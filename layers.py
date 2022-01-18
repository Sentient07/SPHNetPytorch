# layers.py
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from utils import normalized_sh


class SPHConvNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 l_max,
                 nr,
                 patch_size,
                 kernel_radius,
                 strides=0,
                 max_pool = 0,
                 **kwargs):

        super(SPHConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max
        self.nr = nr
        self.patch_size = patch_size
        self.kernel_radius = kernel_radius

        self.strides = strides
        self.max_pool = max_pool
        self.factory_kwargs = {'device': kwargs.get('device', 'cuda'), 'dtype': kwargs.get('dype', 'float32')}
        self.kernel_shape = (self.out_channels, self.in_channels, self.nr, (self.l_max + 1))
        self.weight = Parameter(torch.empty((self.kernel_shape)))
        self.biases = Parameter(torch.empty((out_channels,)))

        self.reset_parameters()
        

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.biases)


    def costly_cdist(self, points, roots):
        # This exists because torch.cdist is inconsistent
        r0 = torch.multiply(roots, roots)
        r0 = torch.sum(r0, dim=2, keepdim=True)
        r1 = torch.multiply(points, points)
        r1 = torch.sum(r1, dim=2, keepdim=True)
        r1 = r1.transpose(2, 1)

        sq_distance_mat = r0 - 2.*torch.matmul(roots, points.transpose(2, 1)) + r1
        return sq_distance_mat


    def compute_patches(self, points, roots):
        """
        Compute point-wise distance matrix, apply KDTree to get 64 neighbour for each
        """
        num_of_points = points.shape[1]
        assert(num_of_points >= self.patch_size)
        # CDist seems inconsistent
        # dist_mat = torch.cdist(points, roots,)
        dist_mat = self.costly_cdist(points, roots)
        sq_patches_dist, patches_idx = torch.topk(-dist_mat, 64*1)
        sq_patches_dist = -sq_patches_dist
        patches = points[torch.arange(points.shape[0]).unsqueeze(1).unsqueeze(2), patches_idx]

        patches = torch.subtract(patches, roots.unsqueeze(2))

        return patches, patches_idx, sq_patches_dist


    def gaussian(self, r, sigma):

        x2 = torch.multiply(r, r)
        return torch.exp(-x2 / (2. * (sigma ** 2)))


    def compute_conv_kernel(self, patches, sq_patches_dist):
        """
        Take the patch, compute its normalised SPH coord,
        compute the gaussian based on patch radius.
        Apply the gaussian on SPH and normalise. 
        """
        Y = normalized_sh(patches, self.l_max)

        # From the patch, compute the radius and Gaussian kernel as func of patch radius
        dist = torch.sqrt(torch.maximum(sq_patches_dist, torch.ones_like(sq_patches_dist)*0.0001))
        dist = dist.unsqueeze(-1)
        r = torch.linspace(start=0., end=self.kernel_radius, steps=self.nr).view(1, 1, 1, self.nr).to(device=dist.device)
        r.requires_grad_(True)
        r = torch.subtract(dist, r)
        sigma = (self.kernel_radius/(self.nr - 1))
        radial_weights = self.gaussian(r, sigma)

        # Apply Gaussian kernel on SPH computed over patch
        radial_weights = radial_weights.unsqueeze(-1)
        y = torch.multiply(Y.unsqueeze(-2), radial_weights)

        y_w = y[:, :, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
        y_w = torch.sum(y_w, dim=2, keepdim=True)
        conv_kernel = torch.divide(y, y_w + 0.000001)
        return conv_kernel


    def forward(self, xyz, signal, return_filters=False):

        # Patches are of size (B, N, P_size, 3)
        patches, patches_idx, sq_patches_dist = self.compute_patches(xyz, xyz)
        conv_kernel = self.compute_conv_kernel(patches, sq_patches_dist)
        # For some reason, the signal is just 1s
        patches = signal[torch.arange(signal.size(0)).unsqueeze(1).unsqueeze(2), patches_idx]

        # Convolve patch with kernel
        y = torch.einsum('bvprn,bvpc->bvcrn', conv_kernel, patches)

        y = torch.multiply(y, y)
        L = []
        p = 0
        for l in range(0, self.l_max + 1):
            x = y[:, :, :, :, p:(p + 2 * l + 1)]
            x = torch.sum(x, dim=-1, keepdim=False)
            L.append(x)
            p += 2 * l + 1

        y = torch.stack(L, dim=-1)
        y = torch.sqrt(torch.maximum(y, torch.ones_like(y)*0.0001))
        y = torch.einsum('ijrn,bvjrn->bvi', self.weight, y)

        # K.bias_add(y, self.biases)
        y += self.biases.unsqueeze(0).unsqueeze(0)

        if return_filters:
            return patches_idx, patches, sq_patches_dist, conv_kernel, y
        return y
