# --------------------------------------------------------
# AS-MLP
# Licensed under The MIT License [see LICENSE for details]
# Written by Zehao Yu and Dongze Lian (AS-MLP)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from model.se import SE3d
from model.voxelization import Voxelization
import functools
import model.functional as VF
from model.shared_mlp import SharedMLP
from model.shared_transformer import SharedTransformer
from lib.pointops.functions import pointops
from torch.autograd import Variable
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden_dim, dim,  1, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AxialShift(nn.Module):
    r""" Axial shift  

    Args:
        dim (int): Number of input channels.
        shift_size (int): shift size .
        as_bias (bool, optional):  If True, add a learnable bias to as mlp. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, shift_size, as_bias=True, proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.conv1 = nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_1 = nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_2 = nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv2_3 = nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.affine_alpha = nn.Parameter(torch.ones([1, dim, 1, 1, 1]))
        self.affine_beta = nn.Parameter(torch.zeros([1, dim, 1, 1, 1]))
        self.conv3 = nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)

        self.actn = nn.GELU()

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)

        # self.shift_dim2 = Shift(self.shift_size, 2)
        # self.shift_dim3 = Shift(self.shift_size, 3)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, R, R, R)

        """
        B, C, R, _, _ = x.shape
        # R = x.shape[-1]

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)

        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(x_pad, self.shift_size, 1)

        def shift(dim):
            x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, R)
            x_cat = torch.narrow(x_cat, 3, self.pad, R)
            x_cat = torch.narrow(x_cat, 4, self.pad, R)

            # # anchor_norm like LocalGroup, 230327 by xgh
            # std = torch.std((x_cat - x).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).\
            #     unsqueeze(dim=-1).unsqueeze(dim=-1)
            # norm_aff = (x_cat - x) / (std + 1e-5)
            # norm_aff = self.affine_alpha * norm_aff + self.affine_beta
            # x_cat = torch.cat([norm_aff, x_cat], dim=1)  # torch.cat([norm_aff, x], dim=1)

            return x_cat

        x_shift_lr = shift(3)
        x_shift_td = shift(2)
        x_shift_hd = shift(4)

        # x_shift_lr = self.shift_dim3(x)
        # x_shift_td = self.shift_dim2(x)
        
        x_lr = self.conv2_1(x_shift_lr)
        x_td = self.conv2_2(x_shift_td)
        x_hd = self.conv2_3(x_shift_hd)

        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)
        x_hd = self.actn(x_hd)

        x = x_lr + x_td + x_hd
        x = self.norm2(x)

        x = self.conv3(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, shift_size={self.shift_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # conv1 
        flops += N * self.dim * self.dim
        # norm 1
        flops += N * self.dim
        # conv2_1 conv2_2
        flops += N * self.dim * self.dim * 2
        # x_lr + x_td
        flops += N * self.dim
        # norm2
        flops += N * self.dim
        # norm3
        flops += N * self.dim * self.dim
        return flops


class AxialShiftedBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        as_bias (bool, optional): If True, add a learnable bias to Axial Mlp. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size=4,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
        #                           ConvPosEnc(dim=dim, k=3, act=cpe_act)])

        # self.norm1 = norm_layer(dim)
        self.norm1 = MyNorm(dim)
        self.axial_shift = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.norm2 = MyNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = FeedForward(dim, dim)

    def forward(self, x):
        B, C, R, _, _ = x.shape

        shortcut = x
        x = self.norm1(x)

        # axial shift block
        x = self.axial_shift(x)  # B, C, R, R, R

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ----------------------------------- copy from PointMLP by xgh 230309
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, reduce, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.reduce = reduce if reduce else None
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = N // self.reduce if self.reduce else N
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        if self.reduce:
            # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
            # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
            fps_idx = pointops.furthestsampling(xyz, S).long()  # (B, N1)
            new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
            new_points = index_points(points, fps_idx)  # [B, npoint, d]
        else:
            new_xyz = xyz.clone()
            new_points = points.clone()

        # print(xyz.size())
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points-mean).reshape(B, -1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)

        if self.reduce is not None:
            return new_xyz, new_points, fps_idx
        else:
            return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class SE1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1)).view(inputs.shape[0], inputs.shape[1], 1)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
                # SE1D(channel, reduction=4),
            )
            # self.net2 = SE1D(channel, reduction=4)

    def forward(self, x):

        return self.act(self.net2(self.net1(x)) + x)


class PointMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduce=4, depths=[1, 1], bias=True,
                 activation='relu', res_expansion=1.0,
                 nsample=32, use_xyz=True
                 ):
        super().__init__()

        self.reduce = reduce
        # group
        self.grouper = LocalGrouper(in_channels, reduce, nsample, use_xyz=use_xyz, normalize='anchor')
        self.group_channels = in_channels*2+3 if use_xyz else in_channels*2

        # PreExtration
        self.transfer = ConvBNReLU1D(self.group_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(depths[0]):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=1, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation0 = nn.Sequential(*operation)

        # PostExtration
        operation = []
        for _ in range(depths[1]):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=1, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation1 = nn.Sequential(*operation)

    def forward(self, input):
        # input: [b, c, n], xyz: [b, 3, n]
        # out: [b, c, n]
        x, xyz = input

        # group
        if self.reduce is not None:
            xyz, x, fps_id = self.grouper(xyz.permute(0, 2, 1), x.permute(0, 2, 1))
        else:
            xyz, x = self.grouper(xyz.permute(0, 2, 1), x.permute(0, 2, 1))


        # Pre
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation0(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)

        # Post
        x = self.operation1(x)

        if self.reduce is not None:
            return x, fps_id
        else:
            return x


# -----------------------------------
import torch.nn as nn
from itertools import repeat


class PointVoxelMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=30, shift_size=5, depth=1, kernel_size=3,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0., reduce=4, use_xyz=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = input_resolution
        self.reduce = reduce
        self.voxelization = Voxelization(self.resolution, normalize=normalize, eps=eps)
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.voxel_encoder = nn.ModuleList([
            AxialShiftedBlock(dim=out_channels, input_resolution=input_resolution,
                              shift_size=shift_size,
                              mlp_ratio=mlp_ratio,
                              as_bias=as_bias,
                              drop=drop,
                              drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                              norm_layer=norm_layer)
            for i in range(depth)])
        self.SE = SE3d(out_channels, reduction=4)
        self.point_features = PointMLPBlock(in_channels, out_channels, reduce=reduce,
                                            res_expansion=1, nsample=32, use_xyz=use_xyz)

        self.global_feats = SharedTransformer(in_channels, out_channels)

        self.vm_feats = None
        self.pm_feats = None
        self.pt_feats = None

    def forward(self, inputs, return_feats=False):
        features, coords = inputs

        # ShiftedVoxelMLP for large agg
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_emb(voxel_features)
        for blk in self.voxel_encoder:
            voxel_features = blk(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = VF.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # PointMLP for local agg
        if self.reduce is not None:
            point_features, fps_id = self.point_features((features, coords))
        else:
            point_features = self.point_features((features, coords))

        # global feats
        global_feats = self.global_feats(features)

        self.vm_feats = voxel_features.detach()
        self.pm_feats = point_features.detach()
        self.pt_feats = global_feats.detach()

        # method1: add
        if return_feats:
            feats = voxel_features + global_feats
        if self.reduce is not None:
            voxel_features = index_points(voxel_features.permute(0, 2, 1), fps_id).permute(0, 2, 1)
            global_feats = index_points(global_feats.permute(0, 2, 1), fps_id).permute(0, 2, 1)
            coords = index_points(coords.permute(0, 2, 1), fps_id).permute(0, 2, 1)
        fused_features = voxel_features + point_features + global_feats

        if return_feats:
            return fused_features, coords, feats
        else:
            return fused_features, coords


class SharedConvBNReLURes1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(out_channels * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(out_channels * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(out_channels * res_expansion), out_channels=out_channels,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(out_channels),
                self.act,
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(out_channels * res_expansion), out_channels=out_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        features, coords = x
        features = self.net1(features)
        features = self.act(self.net2(features) + features)
        return features, coords


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class STNbox(nn.Module):
    def __init__(self, k=6):
        super(STNbox, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points


def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
        # block = SharedConvBNReLURes1D
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


class Upsample(nn.Module):
    def __init__(self, k, in_channels, out_channels, bn_momentum=0.02):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.linear1 = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels))
        # self.linear2 = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels))
        self.linear1 = ConvBNReLU1D(out_channels, out_channels)
        self.linear2 = ConvBNReLU1D(in_channels, out_channels)

    def forward(self, feats, xyz, support_xyz, support_feats=None):
        feats = self.linear1(support_feats) + pointops.interpolation(xyz, support_xyz, self.linear2(feats))
        return feats, support_xyz



# class PartPVMLP(nn.Module):
#
#     def __init__(self, num_classes, num_shapes, in_channel=6, channels=[64, 128, 128, 512, 1024],
#                  resolution=[30, 15, 15, 0, 0], shift_size=[5, 5, 5, 0, 0], groups=1,
#                  res_expansion=1.0, enc_blocks=[1, 1, 1, 1, 1], de_blocks=[1, 1, 1, 1, 1],
#                  reducers=[None, None, None, None, None],
#                  activation="relu", bias=True, use_xyz=True,):
#         super().__init__()
#         self.stages = len(channels)
#         # self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)
#         self.in_channels = in_channel
#         self.num_shapes = num_shapes
#         last_channel = in_channel
#         en_dims = [last_channel]
#         self.enc_list = nn.ModuleList()
#         ### Building Encoder #####
#         concat_channel = 0
#         for i in range(len(enc_blocks)):
#             out_channel = channels[i]
#             if resolution[i] > 1:
#                 layer = PointVoxelMLPBlock(last_channel, out_channel, shift_size=shift_size[i],
#                                            input_resolution=resolution[i], reduce=reducers[i],
#                                            use_xyz=use_xyz)
#             else:
#                 layer = SharedConvBNReLURes1D(last_channel, out_channel)
#             self.enc_list.append(layer)
#             last_channel = out_channel
#             concat_channel += out_channel
#             en_dims.append(last_channel)
#
#         ### Building Decoder #####
#         self.decode_list = nn.ModuleList()
#         en_dims.reverse()
#         # de_dims.insert(0, en_dims[0])
#         for i in range(len(de_blocks)):
#             self.decode_list.append(
#                 PointNetFeaturePropagation(in_channel + en_dims[i], en_dims[i],
#                                            blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
#                                            bias=bias, activation=activation)
#             )
#
#         ### head ###
#         layers, _ = create_mlp_components(
#             in_channels=(num_shapes + last_channel + concat_channel + last_channel),
#             out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
#             classifier=True, dim=2, width_multiplier=1)
#         self.classifier = nn.Sequential(*layers)
#
#         self.stn = STNbox()
#
#     def forward(self, inputs):
#         # inputs : [B, in_channels + S, N]
#         x = inputs[:, :self.in_channels, :]
#         xyz = x[:, :3, :]
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1)
#         one_hot_vectors = inputs[:, -self.num_shapes:, :]
#         num_points = x.size(-1)
#         b = x.size(0)
#
#         # xyz = x[:, :3, :] #.permute(0, 2, 1)
#         xyz_list = [xyz.permute(0, 2, 1)]  # [B, N, 3]
#         x_list = [x]  # [B, D, N]
#         out_x_list = [one_hot_vectors]
#
#         # here is the encoder
#         for i in range(self.stages):
#             if i == 0:
#                 x, xyz, feats_ori = self.enc_list[i]([x, xyz], return_feats=True)
#             else:
#                 x, xyz = self.enc_list[i]([x, xyz])
#             xyz_list.append(xyz.permute(0, 2, 1))
#             x_list.append(x)
#             # out_x_list.append(x)
#
#         # # here is the decoder
#         xyz_list.reverse()
#         x_list.reverse()
#         for i in range(len(self.decode_list)):
#             out_x = self.decode_list[i](xyz_list[-1], xyz_list[i], x_list[-1], x_list[i])
#             # out_x = self.decode_list[i](xyz_list[-1], xyz_list[i], feats_ori, x_list[i])
#             out_x_list.append(out_x)
#         out_x_list.append(x_list[0].max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
#         out_x_list.append(x_list[0].mean(dim=-1, keepdim=True).view(b, -1).unsqueeze(-1).repeat(1, 1, num_points))
#         return self.classifier(torch.cat(out_x_list, dim=1))


class PartPVMLP(nn.Module):

    def __init__(self, args, num_part,  in_channel=6, channels=[64, 128, 128, 512, 1024],
                 resolution=[30, 15, 15, 0, 0], shift_size=[5, 5, 5, 0, 0], groups=1,
                 res_expansion=1.0, enc_blocks=[1, 1, 1, 1, 1], de_blocks=[1, 1, 1, 1, 1],
                 reducers=[None, None, None, None, None],
                 activation="relu", bias=True, use_xyz=True,):
        super().__init__()
        self.stages = len(channels)
        self.num_part = num_part
        # self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)
        self.in_channels = in_channel
        # self.num_shapes = num_shapes
        last_channel = in_channel
        # en_dims = [last_channel]
        self.enc_list = nn.ModuleList()
        ### Building Encoder #####
        concat_channel = 0
        for i in range(len(enc_blocks)):
            out_channel = channels[i]
            if resolution[i] > 1:
                layer = PointVoxelMLPBlock(last_channel, out_channel, shift_size=shift_size[i],
                                           input_resolution=resolution[i], reduce=reducers[i],
                                           use_xyz=use_xyz)
            else:
                layer = SharedConvBNReLURes1D(last_channel, out_channel)
            self.enc_list.append(layer)
            last_channel = out_channel
            concat_channel += out_channel
            # en_dims.append(last_channel)
            
        self.bnt = nn.BatchNorm1d(1024, momentum=0.1)
        self.bnc = nn.BatchNorm1d(64, momentum=0.1)

        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn7 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn8 = nn.BatchNorm1d(128, momentum=0.1)

        self.convt = nn.Sequential(nn.Conv1d(concat_channel, 1024, kernel_size=1, bias=False),
                                   self.bnt)
        self.convc = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bnc)

        self.conv6 = nn.Sequential(nn.Conv1d(1088 + concat_channel, 256, kernel_size=1, bias=False),
                                   self.bn6)
        self.dp1 = nn.Dropout(p=args.get('dropout', 0.4))
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn7)
        self.dp2 = nn.Dropout(p=args.get('dropout', 0.4))
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn8)
        self.conv9 = nn.Conv1d(128, num_part, kernel_size=1, bias=True)



    def forward(self, x, norm_plt, cls_label, gt=None):
        B, C, N = x.size()
        # xyz = x.permute(0, 2, 1)
        xyz = x.clone()
        x = torch.cat([x, norm_plt], dim=1)
        # xyz_list = [xyz]  # [B, N, 3]
        x_list = []  # [B, D, N]
        # here is the encoder
        for i in range(self.stages):
            x, xyz = self.enc_list[i]([x, xyz])
            # xyz_list.append(xyz.permute(0, 2, 1))
            x_list.append(x)

        ###############
        xx = torch.cat(x_list, dim=1)

        xc = F.relu(self.convt(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)

        cls_label = cls_label.view(B, 16, 1)
        cls_label = F.relu(self.convc(cls_label))
        cls = torch.cat((xc.view(B, 1024, 1), cls_label), dim=1)
        cls = cls.repeat(1, 1, N)  # B,1088,N

        x = torch.cat((xx, cls), dim=1)  # 1088+64*3
        x = F.relu(self.conv6(x))
        x = self.dp1(x)
        x = F.relu(self.conv7(x))
        x = self.dp2(x)
        x = F.relu(self.conv8(x))
        x = self.conv9(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # b,n,50

        if gt is not None:
            return x, F.nll_loss(x.contiguous().view(-1, self.num_part), gt.view(-1, 1)[:, 0])
        else:
            return x




def MyNorm(dim):
    return nn.GroupNorm(1, dim)


