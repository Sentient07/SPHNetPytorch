# layers.py
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from torch.nn.parameter import Parameter, UninitializedParameter

import tensorflow as tf

class SPHConvNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 l_max,
                 nr,
                 patch_size,
                 kernel_radius,
                 strides=0,
                 tree_spacing=0,
                 keep_num_points=True,
                 max_pool = 0,
                 initializer='glorot_uniform',
                 l2_regularizer=1.0e-3,
                 with_relu=True, 
                 normalize_patch=False,
                 **kwargs):

        super(SPHConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l_max = l_max
        self.nr = nr
        self.patch_size = patch_size
        self.kernel_radius = kernel_radius

        self.strides = strides
        self.tree_spacing = tree_spacing
        self.keep_num_points = keep_num_points
        self.initializer = initializer
        self.l2_regularizer = l2_regularizer
        self.with_relu = with_relu
        self.max_pool = max_pool
        self.factory_kwargs = {'device': kwargs.get('device', 'cuda'), 'dtype': kwargs.get('dype', 'float32')}
        self.kernel_shape = (self.out_channels, self.in_channels, self.nr, (self.l_max + 1))
        self.weight = Parameter(torch.empty((self.kernel_shape)))
        self.biases = Parameter(torch.empty((out_channels,)))
        self.normalize_patch = normalize_patch

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

        # dist_mat = torch.cdist(points, roots, compute_mode='use_mm_for_euclid_dist_if_necessary')
        dist_mat = self.costly_cdist(points, roots)
        sq_patches_dist, patches_idx = torch.topk(-dist_mat, 64*1)
        sq_patches_dist = -sq_patches_dist
        patches = points[torch.arange(points.shape[0]).unsqueeze(1).unsqueeze(2), patches_idx]

        patches = torch.subtract(patches, roots.unsqueeze(2))

        return patches, patches_idx, sq_patches_dist


    def normalized_sh(self, X_, eps=1e-6):
        """
        Apply L2 normalization to the last layer and then compute SPH on the patch
        """
        # eps is 1e-2 as it's sqrt(1e-4) in TF
        X = nn.functional.normalize(X_, p=2, dim=-1, eps=1e-2)

        assert (4 >= self.l_max >= 1)
        Y = list()
        x = X[..., 0]
        y = X[..., 1]
        z = X[..., 2]

        Y0 = []
        
        Y00 = torch.ones_like(x).float()* (np.sqrt(1. / np.pi) / 2)
        # Y00.requires_grad_(False)
        Y0.append(Y00)

        Y += Y0

        Y1 = []
        Y1_1 = (np.sqrt(3. / np.pi) / 2.) * y
        Y10 = (np.sqrt(3. / np.pi) / 2.) * z
        Y11 = (-np.sqrt(3. / np.pi) / 2.) * x

        Y1.append(Y1_1)
        Y1.append(Y10)
        Y1.append(Y11)

        Y += Y1
        if self.l_max >= 2:
            Y2 = []
            # [x**2, y**2, z**2, x*y, y*z, z*x]
            gathered_X = X[torch.arange(X.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3),
                         torch.arange(X.shape[1]).unsqueeze(1).unsqueeze(2),
                         torch.arange(X.shape[2]).unsqueeze(1),
                         torch.LongTensor([0, 1, 2, 1, 2, 0])]
            X2 = torch.multiply(torch.tile(X, (1, 1, 1, 2)), gathered_X)
            x2 = X2[..., 0]
            y2 = X2[..., 1]
            z2 = X2[..., 2]

            Y2_2 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 3]
            Y2_1 = (np.sqrt(15. / np.pi) / 2.) * X2[..., 4]
            Y20 = (np.sqrt(5. / np.pi) / 4.) * (2. * z2 - x2 - y2)
            Y21 = (-np.sqrt(15. / np.pi) / 2.) * X2[..., 5]
            Y22 = (np.sqrt(15. / np.pi) / 4.) * (x2 - y2)

            Y2.append(Y2_2)
            Y2.append(Y2_1)
            Y2.append(Y20)
            Y2.append(Y21)
            Y2.append(Y22)

            Y += Y2

        if self.l_max >= 3:
            # [x**3, y**3, z**3, x**2*y, y**2*z, z**2*x, x**2*z, y**2*x, z**2*y]
            gathered_X = X[torch.arange(X.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3),
                           torch.arange(X.shape[1]).unsqueeze(1).unsqueeze(2),
                           torch.arange(X.shape[2]).unsqueeze(1),
                           torch.LongTensor([0, 1, 2, 1, 2, 0, 2, 0, 1])]
            X3 = torch.multiply(torch.tile(X2[..., 0:3], (1, 1, 1, 3)), gathered_X)
            xyz = x * y * z

            Y3 = []
            Y3_3 = (np.sqrt(35. / (2. * np.pi)) / 4.) * (3. * X3[..., 3] - X3[..., 1])
            Y3_2 = (np.sqrt(105. / np.pi) / 2.) * xyz
            Y3_1 = (np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., -1] - X3[..., 3] - X3[..., 1])
            Y30 = (np.sqrt(7. / np.pi) / 4.) * (2. * X3[..., 2] - 3. * X3[..., 6] - 3. * X3[..., 4])
            Y31 = (-np.sqrt(21. / (2. * np.pi)) / 4.) * (4. * X3[..., 5] - X3[..., 0] - X3[..., -2])
            Y32 = (np.sqrt(105. / np.pi) / 4.) * (X3[..., -3] - X3[..., 4])
            Y33 = (-np.sqrt(35. / (2. * np.pi)) / 4.) * (X3[..., 0] - 3. * X3[..., -2])

            Y3.append(Y3_3)
            Y3.append(Y3_2)
            Y3.append(Y3_1)
            Y3.append(Y30)
            Y3.append(Y31)
            Y3.append(Y32)
            Y3.append(Y33)

            Y += Y3

        return torch.stack(Y, axis=-1)


    def gaussian(self, r, sigma):

        x2 = torch.multiply(r, r)
        return torch.exp(-x2 / (2. * (sigma ** 2)))


    def compute_conv_kernel(self, patches, sq_patches_dist):
        """
        Take the patch, compute its normalised SPH coord,
        compute the gaussian based on patch radius.
        Apply the gaussian on SPH and normalise. 
        """
        Y = self.normalized_sh(patches, eps=0.0001)

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
