import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import torch

def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob('./data/modelnet40_ply_hdf5_2048/ply_data_%s*.h5' % partition):
        f = h5py.File(h5_name, mode='r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', pt_norm=False):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.pt_norm = pt_norm

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            if self.pt_norm:
                pointcloud = pc_normalize(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)  # shuffle the order of pts
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids



# class ModelNetDataLoader(Dataset):
#     def __init__(self, npoint=1024, partition='train', uniform=False, normal_channel=True, cache_size=15000):
#         # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#         # DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')
#         DATA_DIR = os.path.join('./data', 'modelnet40_normal_resampled')
#
#         self.npoints = npoint
#         self.uniform = uniform
#         self.catfile = os.path.join(DATA_DIR, 'modelnet40_shape_names.txt')
#
#         self.cat = [line.rstrip() for line in open(self.catfile)]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#         self.normal_channel = normal_channel
#
#         shape_ids = {}
#         shape_ids['train'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_train.txt'))]
#         shape_ids['test'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_test.txt'))]
#
#         assert (partition == 'train' or partition == 'test')
#         shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[partition]]
#         # list of (shape_name, shape_txt_file_path) tuple
#         self.datapath = [(shape_names[i], os.path.join(DATA_DIR, shape_names[i], shape_ids[partition][i]) + '.txt') for i
#                          in range(len(shape_ids[partition]))]
#         print('The size of %s data is %d'%(partition,len(self.datapath)))
#
#         self.cache_size = cache_size  # how many data points to cache in memory
#         self.cache = {}  # from index to (point_set, cls) tuple
#
#     def __len__(self):
#         return len(self.datapath)
#
#     def _get_item(self, index):
#         if index in self.cache:
#             point_set, cls = self.cache[index]
#         else:
#             fn = self.datapath[index]
#             cls = self.classes[self.datapath[index][0]]
#             cls = np.array([cls]).astype(np.int32)
#             point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
#             if self.uniform:
#                 point_set = farthest_point_sample(point_set, self.npoints)
#             else:
#                 point_set = point_set[0:self.npoints,:]
#
#             point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#
#             if not self.normal_channel:
#                 point_set = point_set[:, 0:3]
#
#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls)
#
#         return point_set, cls
#
#     def __getitem__(self, index):
#         return self._get_item(index)



class ModelNetDataLoader(Dataset):
    def __init__(self, npoint=1024, partition='train', uniform=False, normal_channel=True, cache_size=15000):
        # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')
        DATA_DIR = os.path.join('./data', 'modelnet40_normal_resampled')

        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(DATA_DIR, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(DATA_DIR, 'modelnet40_test.txt'))]

        assert (partition == 'train' or partition == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[partition]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(DATA_DIR, shape_names[i], shape_ids[partition][i]) + '.txt') for i
                         in range(len(shape_ids[partition]))]
        print('The size of %s data is %d'%(partition,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)