from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from timm.scheduler import CosineLRScheduler
from data import _ShapeNetDataset
from model.pvmlp import PartPVMLP
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, AverageMeter
import sklearn.metrics as metrics
import time


seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    os.system('cp main_part_seg.py checkpoints' + '/' + args.exp_name + '/' + 'main_part_seg.py.backup')
    os.system('cp model/pvmlp.py checkpoints' + '/' + args.exp_name + '/' + 'part_pvmlp.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]-1]
            num = seg_num[label[shape_idx]-1]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def train(args, io):
    train_dataset = _ShapeNetDataset(num_points=args.num_points, partition='trainval')
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=16, batch_size=args.batch_size, shuffle=True,
                              drop_last=drop_last)
    test_loader = DataLoader(_ShapeNetDataset(num_points=args.num_points, partition='test'),
                             num_workers=16, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    num_classes = train_loader.dataset.num_classes
    num_shapes = train_loader.dataset.num_shapes
    if args.model == 'psvmlp':
        # model = pvt_partseg(num_classes=num_classes, num_shapes=num_shapes).to(device)
        model = PartPVMLP(num_classes=num_classes, num_shapes=num_shapes).to(device)

    else:
        raise Exception("Not implemented")

    # model = nn.DataParallel(model)
    io.cprint("model parameters: %.0f" % sum(param.numel() for param in model.parameters()))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.opt == 'SGD':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    elif args.opt == 'Adam':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'AdamW':
        print("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)
    elif args.scheduler == 'cosLR':
        scheduler = CosineLRScheduler(opt,
                                      t_initial=args.epochs,
                                      cycle_mul=1,
                                      lr_min=1e-6,
                                      cycle_decay=0.1,
                                      warmup_lr_init=1e-6,
                                      warmup_t=args.warmup_epochs,
                                      cycle_limit=1,
                                      t_in_epochs=True)

    criterion = cal_loss

    best_test_iou = 0
    MB = 1024.0 * 1024.0
    batch_time = AverageMeter()
    max_iter = args.epochs * len(train_loader)
    end = time.time()
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for i, (data, seg, shape_label) in enumerate(train_loader):
            batch_size = data.size()[0]
            data, seg = data.to(device), seg.to(device)
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, num_classes), seg.view(-1, 1).squeeze())
            loss.backward()
            opt.step()

            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            train_true_cls.append(seg_np.reshape(-1))
            train_pred_cls.append(pred_np.reshape(-1))
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(shape_label)

            # calculate remain time
            batch_time.update(time.time() - end)
            end = time.time()
            current_iter = epoch * len(train_loader) + i + 1
            remain_iter = max_iter - current_iter
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        elif args.scheduler == 'cosLR':
            scheduler.step(epoch)

        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        memory = torch.cuda.max_memory_allocated() / MB
        outstr = 'train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train ins iou: %.6f, ' \
                 'mem: %d, remain_time: %s' \
                 % (epoch, train_loss * 1.0 / count, train_acc, avg_per_class_acc, np.mean(train_ious),
                    memory, remain_time)
        io.cprint(outstr)


        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, seg, shape_label in test_loader:
            data, seg = data.to(device), seg.to(device)
            batch_size = data.size()[0]
            with torch.no_grad():
                seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, num_classes), seg.view(-1, 1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(shape_label)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test ins iou: %.6f' % (epoch,
                                                                                              test_loss * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'checkpoints/%s/partmodel.t7' % args.exp_name)
            io.cprint('Max ins iou: %.6f' % best_test_iou)
    io.cprint('EXP FINISHED!, Max ins iou: %.6f' % best_test_iou)


def test(args, io):
    test_loader = DataLoader(_ShapeNetDataset(num_points=args.num_points, partition='test'),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    num_classes = test_loader.dataset.num_classes
    num_shapes = test_loader.dataset.num_shapes

    # Try to load models
    if args.model == 'pvmlp':
        model = PartPVMLP(num_classes=num_classes, num_shapes=num_shapes).to(device)
    else:
        raise Exception("Not implemented")

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, seg, shape_label in test_loader:
        data, seg = data.to(device), seg.to(device)
        seg_pred = model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(shape_label)
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious))
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='partseg_pvmlp_230411_AdamWEpoch350', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='psvmlp', metavar='N',
                        choices=['psvmlp'],
                        help='Model to use, [pvt]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train ')
    # parser.add_argument('--use_sgd', type=bool, default=False,
    #                     help='Use SGD')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'],
                        help='Choose opt')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='warmup epochs (default: 10, 10 if epochs is 300)')
    parser.add_argument('--weight_decay', type=float, default=0.05, metavar='LR',
                        help='weight decay (default: 0.05, 0.05 if using adamw)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step', 'cosLR'],
                        help='Scheduler to use, [cos, step, cosLR]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='checkpoints/partseg/partmodel.t7', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/partrun.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
