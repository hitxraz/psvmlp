""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
# import cv2
import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
matplotlib.use('TkAgg')


def vis_pred(xyz, gt=None, pred=None, label=None):
    cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                     [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])
    # seg = gt.cpu().data.numpy()
    seg = gt
    seg = seg - seg.min()
    c_gt = cmap[seg, :]

    seg_pred = pred.max(dim=2)[1]  # (batch_size, num_points)  the pred_class_idx of each point in each sample
    # seg_pred = seg_pred.cpu().data.numpy()
    c_pred = cmap[seg_pred, :]

    # color
    c0 = c_gt[:, 0]
    c1 = c_gt[:, 1]
    c2 = c_gt[:, 2]
    # normalize
    c0 /= (c0.max() + 1e-14)
    c1 /= (c1.max() + 1e-14)
    c2 /= (c2.max() + 1e-14)
    c0 = np.require(c0, 'float32', 'C')[:, None]
    c1 = np.require(c1, 'float32', 'C')[:, None]
    c2 = np.require(c2, 'float32', 'C')[:, None]
    c_gt_lst = np.concatenate([c0, c1, c2], axis=1)

    c0 = c_pred[:, 0]
    c1 = c_pred[:, 1]
    c2 = c_pred[:, 2]
    c0 /= (c0.max() + 1e-14)
    c1 /= (c1.max() + 1e-14)
    c2 /= (c2.max() + 1e-14)
    c0 = np.require(c0, 'float32', 'C')[:, None]
    c1 = np.require(c1, 'float32', 'C')[:, None]
    c2 = np.require(c2, 'float32', 'C')[:, None]
    c_pred_lst = np.concatenate([c0, c1, c2], axis=1)

    # Creating plot
    fig = plt.figure(figsize=(5, 10))
    ax = plt.subplot(211, projection="3d")
    ax.grid(None)
    ax.axis('off')
    # 创建x轴定位器，间隔2
    x_major_locator = MultipleLocator(0.1)
    # 创建y轴定位器，间隔5
    y_major_locator = MultipleLocator(0.1)
    z_major_locator = MultipleLocator(0.1)
    # 获取轴对象
    axes = plt.gca()
    # 设置x轴的间隔
    axes.xaxis.set_major_locator(x_major_locator)
    # 设置y轴的间隔
    axes.yaxis.set_major_locator(y_major_locator)
    axes.zaxis.set_major_locator(z_major_locator)
    # ax.view_init(25, 20, 90)
    ax.view_init(20, 20, 90)
    sctt_gt = ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=1, c=c_gt_lst,)

    ax1 = plt.subplot(212, projection="3d")
    ax1.grid(None)
    ax1.axis('off')
    # 创建x轴定位器，间隔2
    x_major_locator = MultipleLocator(0.1)
    # 创建y轴定位器，间隔5
    y_major_locator = MultipleLocator(0.1)
    z_major_locator = MultipleLocator(0.1)
    # 获取轴对象
    axes = plt.gca()
    # 设置x轴的间隔
    axes.xaxis.set_major_locator(x_major_locator)
    # 设置y轴的间隔
    axes.yaxis.set_major_locator(y_major_locator)
    axes.zaxis.set_major_locator(z_major_locator)
    ax1.view_init(20, 20, 90)
    sctt_pred = ax1.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=1, c=c_pred_lst, )

    # show plot
    plt.savefig('{}_{}.png'.format(label, time.time()), dpi=1000)

if __name__ == '__main__':
    import os
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../data/shapenet', help='dataset path')
    parser.add_argument('--category', type=str, default='Airplane', help='select category')
    parser.add_argument('--npoints', type=int, default=2048, help='resample points number')
    parser.add_argument('--ballradius', type=int, default=10, help='ballradius')
    opt = parser.parse_args()
    '''
    Airplane	02691156
    Bag	        02773838
    Cap	        02954340
    Car	        02958343
    Chair	    03001627
    Earphone	03261776
    Guitar	    03467517
    Knife	    03624134
    Lamp	    03636649
    Laptop	    03642806
    Motorbike   03790512
    Mug	        03797390
    Pistol	    03948459
    Rocket	    04099429
    Skateboard  04225987
    Table	    04379243'''

    cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                     [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

    # from ShapeNetDataLoader import PartNormalDataset

    from data_util import PartNormalDataset
    root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    dataset = PartNormalDataset(npoints=2048, split='test', normalize=False)
    idx = np.random.randint(0, len(dataset))
    data = dataset[idx]
    point_set, _, seg, _ = data
    choice = np.random.choice(point_set.shape[0], opt.npoints, replace=True)
    point_set, seg = point_set[choice, :], seg[choice]
    seg = seg - seg.min()
    gt = cmap[seg, :]

    # seg_pred = model(points, norm_plt, to_categorical(label, num_classes))
    # seg_pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)  the pred_class_idx of each point in each sample
    # seg_pred = seg_pred.cpu().data.numpy()
    #
    # pred_c = cmap[seg_pred, :]

    # Import libraries
    from mpl_toolkits import mplot3d
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # Creating dataset
    # z = 4 * np.tan(np.random.randint(10, size=(500))) + np.random.randint(100, size=(500))
    # x = 4 * np.cos(z) + np.random.normal(size=500)
    # y = 4 * np.sin(z) + 4 * np.random.normal(size=500)

    # Creating figure
    fig = plt.figure(figsize=(5, 10))
    ax = plt.subplot(211, projection="3d")
    ax.grid(None)
    ax.axis('off')
    # 创建x轴定位器，间隔2
    x_major_locator = MultipleLocator(0.1)
    # 创建y轴定位器，间隔5
    y_major_locator = MultipleLocator(0.1)
    z_major_locator = MultipleLocator(0.1)

    # 获取轴对象
    axes = plt.gca()
    # 设置x轴的间隔
    axes.xaxis.set_major_locator(x_major_locator)
    # 设置y轴的间隔
    axes.yaxis.set_major_locator(y_major_locator)
    axes.zaxis.set_major_locator(z_major_locator)

    ax.view_init(20, 20, 90)

    # # Add x, y gridlines
    # ax.grid(b=True, color='grey',
    #         linestyle='-.', linewidth=0.3,
    #         alpha=0.2)

    # Creating color map
    # my_cmap = plt.get_cmap('hsv')

    # color
    c0 = gt[:, 0]
    c1 = gt[:, 1]
    c2 = gt[:, 2]

    c0 /= (c0.max() + 1e-14) / 255.0
    c1 /= (c1.max() + 1e-14) / 255.0
    c2 /= (c2.max() + 1e-14) / 255.0

    c0 /= 255.0
    c1 /= 255.0
    c2 /= 255.0

    c0 = np.require(c0, 'float32', 'C')[:, None]
    c1 = np.require(c1, 'float32', 'C')[:, None]
    c2 = np.require(c2, 'float32', 'C')[:, None]

    color_lst = np.concatenate([c0, c1, c2], axis=1)

    # Creating plot
    sctt = ax.scatter3D(point_set[:, 0], point_set[:, 1], point_set[:, 2],
                        alpha=1,
                        s=1,
                        c=color_lst,
                        )

    # plt.title("simple 3D scatter plot")
    # ax.set_xlabel('X-axis', fontweight='bold')
    # ax.set_ylabel('Y-axis', fontweight='bold')
    # ax.set_zlabel('Z-axis', fontweight='bold')
    # fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)

    # show plot
    plt.savefig('0.png', dpi=1000)
    plt.show()

    # showpoints(point_set, gt, c_pred=pred, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
    #            background=(255, 255, 255), normalizecolor=True, ballradius=opt.ballradius)


