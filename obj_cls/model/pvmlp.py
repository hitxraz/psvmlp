# --------------------------------------------------------
# AS-MLP
# Licensed under The MIT License [see LICENSE for details]
# Written by Zehao Yu and Dongze Lian (AS-MLP)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from .shift_cuda import Shift
from model.se import SE3d
from model.voxelization import Voxelization
import functools
import model.functional as VF
from model.shared_mlp import SharedMLP
#from util.PAConv_util import get_scorenet_input, knn, feat_trans_dgcnn, ScoreNet
#from cuda_lib.functional import assign_score_withk as assemble_dgcnn
from model.shared_transformer import SharedTransformer
from typing import Tuple
from lib.pointops.functions import pointops




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

    
    def shift(self, dim, x):
        R = x.shape[-1]
        #x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "reflect")
        xs = torch.chunk(x_pad, self.shift_size, 1)
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

        #x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        #xs = torch.chunk(x_pad, self.shift_size, 1)

        def shift0(dim, xx):
            x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xx, range(-self.pad, self.pad+1))]
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

        #x_shift_lr = shift(3)
        #x_shift_td = shift(2)
        #x_shift_hd = shift(4)
        x = self.shift(3, x)
        x = self.conv2_2(x)
        x = self.shift(2, x)
        x = self.conv2_1(x)
        x = self.shift(4, x)
        x = self.conv2_3(x)
        
        

        x = self.actn(x)
        x = self.norm2(x)

        
        #x_lr = self.conv2_1(x_shift_lr)
        #x_td = self.conv2_2(x_shift_td)
        #x_hd = self.conv2_3(x_shift_hd)

        #x_lr = self.actn(x_lr)
        #x_td = self.actn(x_td)
        #x_hd = self.actn(x_hd)

        #x = x_lr + x_td + x_hd
        #x = self.norm2(x)

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



class VoxelMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=30, shift_size=5, depth=2, kernel_size=3,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size
        self.resolution = input_resolution
        self.boxsize = 3
        # self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.2
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
        # self.SE = SE3d(out_channels)
        # self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_emb(voxel_features)
        for blk in self.voxel_encoder:
            voxel_features = blk(voxel_features)
        # voxel_features = self.voxel_encoder(voxel_features)
        # voxel_features = self.SE(voxel_features)
        voxel_features = VF.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features #+ self.point_features(features)
        return fused_features, coords


class VoxelMLPTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=30, shift_size=5, depth=2, kernel_size=3,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size
        self.resolution = input_resolution
        # self.boxsize = 3
        # self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.2
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
        self.SE = SE3d(out_channels, reduction=2)
        # # self.SA = sa3d_layer(out_channels, groups=64)
        # self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_emb(voxel_features)
        for blk in self.voxel_encoder:
            voxel_features = blk(voxel_features)
        voxel_features = self.SE(voxel_features)
        # voxel_features = self.SA(voxel_features)
        voxel_features = VF.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # pos = coords.permute(0, 2, 1)
        # rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        # rel_pos = rel_pos.sum(dim=-1)

        fused_features = voxel_features #+ self.point_features(features)
        return fused_features, coords



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
    def __init__(self, in_channels, out_channels, reduce=None, depths=[1, 1], bias=True,
                 activation='relu',
                 nsample=24, use_xyz=False
                 ):
        super().__init__()

        # group
        self.grouper = LocalGrouper(in_channels, reduce, nsample, use_xyz=False, normalize='anchor')
        self.group_channels = in_channels*2+3 if use_xyz else in_channels*2

        # PreExtration
        self.transfer = ConvBNReLU1D(self.group_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(depths[0]):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=1, res_expansion=1,
                                bias=bias, activation=activation)
            )
        self.operation0 = nn.Sequential(*operation)

        # PostExtration
        operation = []
        for _ in range(depths[1]):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=1, res_expansion=1, bias=bias, activation=activation)
            )
        self.operation1 = nn.Sequential(*operation)

    def forward(self, input):
        # input: [b, c, n], xyz: [b, 3, n]
        # out: [b, c, n]
        x, xyz = input

        # group
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

        return x



class PointVoxelMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=30, shift_size=5, depth=1, kernel_size=3,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = input_resolution
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
        self.point_features = PointMLPBlock(in_channels, out_channels)

        self.global_feats = SharedTransformer(in_channels, out_channels)


    def forward(self, inputs):
        features, coords = inputs

        # ShiftedVoxelMLP for large agg
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_emb(voxel_features)
        for blk in self.voxel_encoder:
            voxel_features = blk(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = VF.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # PointMLP for local agg
        point_features = self.point_features((features, coords))  # 230530

        # global feats
        global_feats = self.global_feats(features)  # 230530 

        # method1: add
        fused_features = voxel_features + point_features + global_feats
       

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
                # SE1D(out_channels, reduction=4)
            )
            # self.net2 = SE1D(out_channels, reduction=4)

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


class PVMLP(nn.Module):
  
    blocks = ((64, 1, 30), (128, 2, 15), (512, 1, 0), (1024, 1, 0))  # pvmlp230321 cnrr

    def __init__(self, in_channels=3, num_classes=40, width_multiplier=1, voxel_resolution_multiplier=1,
                 activation="relu", bias=False, drop_path_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        # self.embedding = ConvBNReLU1D(in_channels, embed_dim, bias=bias, activation=activation)
        # self.spd = SpatialDropout(0.2)

        # # 230402 by xgh 93.56
        # self.emb = ShiftedPointEmbedding(emb_dim, num_vote=10, with_norm=True)

        layers, channels_point, concat_channels_point = self._make_layers(
            blocks=self.blocks, in_channels=in_channels, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(channels_point + concat_channels_point + channels_point, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024))

        self.linear1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def _make_layers(self, blocks, in_channels, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
        r, vr = width_multiplier, voxel_resolution_multiplier

        layers, concat_channels = [], 0
        for out_channels, num_blocks, voxel_resolution in blocks:
            out_channels = int(r * out_channels)
            if voxel_resolution > 0:
                block = functools.partial(PointVoxelMLPBlock, kernel_size=3,
                                          input_resolution=int(vr * voxel_resolution), shift_size=3, #230531 shift 3,5,7
                                          normalize=normalize, eps=eps, depth=1)
            elif voxel_resolution < 0:
                block = functools.partial(DGCNN_PAConv, k=20, m1=8, calc_scores='softmax')  # 93.35
            else:
                block = functools.partial(SharedConvBNReLURes1D, res_expansion=1)
                # block = SharedMLP
            for _ in range(num_blocks):
                layers.append(block(in_channels, out_channels))
                in_channels = out_channels
                concat_channels += out_channels
               
        return layers, in_channels, concat_channels

    def forward(self, features, label=None, criterion=None):
        # inputs : xyz[B, 3, N]
        num_points, batch_size = features.size(-1), features.size(0)

        coords = features[:, :3, :]

        # features = self.emb(features)
        # features = self.spd(features)

        out_features_list = []
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)

        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, num_points]))
        out_features_list.append(
            features.mean(dim=-1, keepdim=True).view(batch_size, -1).unsqueeze(-1).repeat(1, 1, num_points))

        features = torch.cat(out_features_list, dim=1)
        features = F.leaky_relu(self.conv_fuse(features))
        features = F.adaptive_max_pool1d(features, 1).view(batch_size, -1)
        features = F.leaky_relu(self.bn1(self.linear1(features)))
        features = self.dp1(features)
        features = F.leaky_relu(self.bn2(self.linear2(features)))
        features = self.dp2(features)
        features = self.linear3(features)
        
        if criterion is not None:
            return features, criterion(features, label)  # return output and loss
        else:
            return features
        #return self.linear3(features)





def MyNorm(dim):
    return nn.GroupNorm(1, dim)


