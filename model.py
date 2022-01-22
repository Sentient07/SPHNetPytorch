import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import SPHConvNet


class PointNetWithSPH(nn.Module):
    def __init__(self, out_channels, kernel_radius, strides, l_max, nr, patch_size,
                 kd_pool_ratio, nlatent, num_inp=3):
        """
        Encoder
        """

        super(PointNetWithSPH, self).__init__()
        self.conv1 = SPHConvNet(num_inp, out_channels[0], l_max, nr, patch_size, kernel_radius[0], 
                                strides=strides[0], tree_spacing=0, keep_num_points=True,
                                max_pool = 0, l2_regularizer=1.0e-3, with_relu=True,  
                                normalize_patch=False)
        self.conv2 = SPHConvNet(out_channels[0], out_channels[1], l_max, nr, patch_size, kernel_radius[1], 
                                strides=strides[1], tree_spacing=0, keep_num_points=True,
                                max_pool = 0, l2_regularizer=1.0e-3, with_relu=True,  
                                normalize_patch=False)
        self.conv3 = SPHConvNet(out_channels[1], out_channels[2], l_max, nr, patch_size, kernel_radius[2], 
                                strides=strides[2], tree_spacing=0, keep_num_points=True,
                                max_pool = 0, l2_regularizer=1.0e-3, with_relu=True,  
                                normalize_patch=False)
        
        self.kdTreePool_1 = nn.MaxPool1d((4, ), stride=(4, ))
        self.kdTreePool_2 = nn.MaxPool1d((4, ), stride=(4, ))

        self.pool_window = int(np.rint(np.log(kd_pool_ratio)/np.log(2.)))
        self.lin1 = nn.Linear(nlatent, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 40)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        #  track_running_stats=False because xyz and invariant signal are from diff distribution.
        self.bn1 = torch.nn.BatchNorm1d(out_channels[0], track_running_stats=False, momentum=0.5)
        self.bn2 = torch.nn.BatchNorm1d(out_channels[1], track_running_stats=False, momentum=0.5)
        self.bn3 = torch.nn.BatchNorm1d(out_channels[2], track_running_stats=False, momentum=0.5)
        self.bn4 = torch.nn.BatchNorm1d(512, track_running_stats=False, momentum=0.5)
        self.bn5 = torch.nn.BatchNorm1d(256, track_running_stats=False, momentum=0.5)

        self.nlatent = nlatent

    def forward(self, inp_pc):
        inp_pc = inp_pc.view(inp_pc.shape[0], -1, 3)
        second_signal = torch.ones((inp_pc.shape[0], inp_pc.shape[1], 1)).to(inp_pc.device).float()
        x = self.conv1(inp_pc, second_signal, return_filters=False)
        x = self.kdTreePool_1(F.relu(self.bn1(x.transpose(1,2)))).transpose(1,2)
        inp_downsampled = F.avg_pool1d(inp_pc.transpose(1,2), (4, ), (4, )).transpose(1,2)
        x = self.kdTreePool_2(F.relu(self.bn2(self.conv2(inp_downsampled, x,
                                                         return_filters=False).transpose(1,2)))).transpose(1,2)
        inp_downsampled = F.avg_pool1d(inp_downsampled.transpose(1,2), (4, ), (4, )).transpose(1,2)
        x = F.relu(self.bn3(self.conv3(inp_downsampled, x, return_filters=False).transpose(1,2))).transpose(1,2)
        x, _ = torch.max(x, 1)
        x = self.drop1(F.relu(self.bn4(self.lin1(x))))
        x = self.drop2(F.relu(self.bn5(self.lin2(x))))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x