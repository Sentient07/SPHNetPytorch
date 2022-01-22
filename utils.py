# utils.py
import torch
import torch.nn as nn
import trimesh
import numpy as np

def normalized_sh(X_, l_max=3):
    """
    Apply L2 normalization to the last layer and then compute SPH on the patch.
    Handmade function. Is it possible to do it more simply? Any packages? TODO
    """
    # eps is 1e-2 as it's sqrt(1e-4) in TF
    X = nn.functional.normalize(X_, p=2, dim=-1, eps=1e-2)

    assert (4 > l_max >= 1)
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
    if l_max >= 2:
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

    if l_max >= 3:
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


def save_xyz(pts, file_name):
    s = trimesh.util.array_to_string(pts)
    with open(file_name, 'w') as f:
        f.write("%s\n" % s)