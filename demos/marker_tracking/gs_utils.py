import os
import numpy as np
import math
import struct
import cv2
import open3d
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack
from scipy.interpolate import griddata


class Header():
    def __init__(self):
        self.imW = None
        self.imH = None
        self.lengthx = None
        self.lengthy = None
        self.offsetx = None
        self.offsety = None
        self.mmpp = None

    def __str__(self):
        return str(self.__class__) + '\n' + '\n'.join(
            ('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))


def read_tmd(fpath):
    heightMap = None
    headerData = None
    if not os.path.exists(fpath) or not fpath.lower().endswith('.tmd'):
        return heightMap, headerData

    # print('**********')
    header_len = 32
    comment_len = 2048
    int32_len = 4
    float_len = 4

    with open(fpath, 'rb') as f:
        tmd = f.read()

        headerData = Header()

        start = 0
        end = start + header_len
        TMD_HEADER = tmd[start:end].decode('UTF-8')
        # print(TMD_HEADER)

        start = end
        end += comment_len
        # commentbuffer = tmd[start:end]
        # print(tmd[start:end+100])

        while start < end:
            if tmd[start] == 0:
                end = start + 1
                break
            start += 1

        start = end
        end += int32_len
        imW_byte = tmd[start:end]
        headerData.imW = struct.unpack('i', imW_byte)[0]

        start = end
        end += int32_len
        imH_byte = tmd[start:end]
        headerData.imH = struct.unpack('i', imH_byte)[0]

        start = end
        end += float_len
        lengthx_byte = tmd[start:end]
        headerData.lengthx = struct.unpack('f', lengthx_byte)[0]

        start = end
        end += float_len
        lengthy_byte = tmd[start:end]
        headerData.lengthy = struct.unpack('f', lengthy_byte)[0]

        start = end
        end += float_len
        offsetx_byte = tmd[start:end]
        headerData.offsetx = struct.unpack('f', offsetx_byte)[0]

        start = end
        end += float_len
        offsety_byte = tmd[start:end]
        headerData.offsety = struct.unpack('f', offsety_byte)[0]

        headerData.mmpp = headerData.lengthx / headerData.imW

        pxOffX = int(headerData.offsetx / headerData.mmpp)
        pxOffY = int(headerData.offsety / headerData.mmpp)
        fullW = headerData.imW + pxOffX
        fullH = headerData.imH + pxOffY

        heightMap = np.zeros((int(fullH), int(fullW)), dtype=np.float32)
        # heightMap = np.zeros((headerData.imH, headerData.imW), dtype=np.float32)
        for y in range(headerData.imH):
            start = end
            end += float_len * headerData.imW
            heightMap[y + pxOffY:y + pxOffY + 1][pxOffX:] = struct.unpack('f' * headerData.imW, tmd[start:end])

    return heightMap, headerData


def write_tmd(fpath, heightMap, mmpp):
    pdir = os.path.dirname(os.path.abspath(fpath))
    if not os.path.exists(pdir) or not fpath.lower().endswith('.tmd'):
        return None

    cols = heightMap.shape[1]
    rows = heightMap.shape[0]

    with open(fpath, "wb") as fd:
        fd.write('Binary TrueMap Data File v2.0'.encode())
        fd.write('\r'.encode())
        fd.write('\n'.encode())
        fd.write(bytearray([0]))
        fd.write(bytearray([0]))

        # image size
        fd.write(struct.pack('i', cols))
        fd.write(struct.pack('i', rows))

        # length and width of axes
        fd.write(struct.pack('f', mmpp * cols))
        fd.write(struct.pack('f', mmpp * rows))

        # offset
        fd.write(struct.pack('f', mmpp * 0.0))
        fd.write(struct.pack('f', mmpp * 0.0))

        # Write matrix
        for y in range(rows):
            rdata = heightMap[y, :]
            fd.write(struct.pack('f' * len(rdata), *rdata))

    return True


def dilate(img, ksize=5, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)


def interpolate_grad(img, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # pixel around markers
    mask_around = (dilate(mask, ksize=3, iter=2) > 0) & ~(mask != 0)
    # mask_around = mask == 0
    mask_around = mask_around.astype(np.uint8)
    #cv2.imshow("mask_around", mask_around*1.)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    # mask_zero = mask == 0
    mask_zero = mask_around == 1
    #cv2.imshow("mask_zero", mask_zero*1.)

    # if np.where(mask_zero)[0].shape[0] != 0:
    #     print ('interpolating')
    mask_x = xx[mask_around == 1]
    mask_y = yy[mask_around == 1]
    points = np.vstack([mask_x, mask_y]).T
    values = img[mask_x, mask_y]
    markers_points = np.vstack([xx[mask != 0], yy[mask != 0]]).T
    method = "nearest"
    # method = "linear"
    # method = "cubic"
    x_interp = griddata(points, values, markers_points, method=method)
    x_interp[x_interp != x_interp] = 0.0
    ret = img.copy()
    ret[mask != 0] = x_interp

    return ret


def demark(gx, gy, markermask):
    # mask = find_marker(img)
    gx_interp = interpolate_grad(gx.copy(), markermask)
    gy_interp = interpolate_grad(gy.copy(), markermask)
    return gx_interp, gy_interp


class RGB2NormNet(nn.Module):
    def __init__(self):
        super(RGB2NormNet, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        return x


def poisson_dct_neumaan(gx, gy):
    gxx = 1 * (gy[(list(range(1, gx.shape[0])) + [gx.shape[0] - 1]), :] - gy[([0] + list(range(gx.shape[0] - 1))), :])
    gyy = 1 * (gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])] - gx[:, ([0] + list(range(gx.shape[1] - 1)))])
    f = gxx + gyy

    ### Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    b[0, 1:-2] = -gy[0, 1:-2]
    b[-1, 1:-2] = gy[-1, 1:-2]
    b[1:-2, 0] = -gx[1:-2, 0]
    b[1:-2, -1] = gx[1:-2, -1]
    b[0, 0] = (1 / np.sqrt(2)) * (-gy[0, 0] - gx[0, 0])
    b[0, -1] = (1 / np.sqrt(2)) * (-gy[0, -1] + gx[0, -1])
    b[-1, -1] = (1 / np.sqrt(2)) * (gy[-1, -1] + gx[-1, -1])
    b[-1, 0] = (1 / np.sqrt(2)) * (gy[-1, 0] - gx[-1, 0])

    ## Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0, 1:-2] = f[0, 1:-2] - b[0, 1:-2]
    f[-1, 1:-2] = f[-1, 1:-2] - b[-1, 1:-2]
    f[1:-2, 0] = f[1:-2, 0] - b[1:-2, 0]
    f[1:-2, -1] = f[1:-2, -1] - b[1:-2, -1]

    ## Modification near the corners (Eq. 54 in [1])
    f[0, -1] = f[0, -1] - np.sqrt(2) * b[0, -1]
    f[-1, -1] = f[-1, -1] - np.sqrt(2) * b[-1, -1]
    f[-1, 0] = f[-1, 0] - np.sqrt(2) * b[-1, 0]
    f[0, 0] = f[0, 0] - np.sqrt(2) * b[0, 0]

    ## Cosine transform of f
    tt = fftpack.dct(f, norm='ortho')
    fcos = fftpack.dct(tt.T, norm='ortho').T

    # Cosine transform of z (Eq. 55 in [1])
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * ((np.sin(0.5 * math.pi * x / (f.shape[1]))) ** 2 + (np.sin(0.5 * math.pi * y / (f.shape[0]))) ** 2)

    # 4 * ((sin(0.5 * pi * x / size(p, 2))). ^ 2 + (sin(0.5 * pi * y / size(p, 1))). ^ 2)

    f = -fcos / denom
    # Inverse Discrete cosine Transform
    tt = fftpack.idct(f, norm='ortho')
    img_tt = fftpack.idct(tt.T, norm='ortho').T

    img_tt = img_tt.mean() + img_tt
    # img_tt = img_tt - img_tt.min()

    return img_tt


class Visualize3D:
    def __init__(self, n, m, mmpp, pc_savepath):
        self.n, self.m = n, m
        self.init_open3D(mmpp)
        self.cnt = 212
        self.mmpp = mmpp
        self.save_path = pc_savepath
        pass

    def init_open3D(self, mmpp):
        x = np.arange(self.n) * mmpp
        y = np.arange(self.m) * mmpp
        self.X, self.Y = np.meshgrid(x, y)
        Z = np.sin(self.X)

        self.points = np.zeros([self.n * self.m, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X)
        self.points[:, 1] = np.ndarray.flatten(self.Y)

        self.depth2points(Z)

        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)

    def update(self, Z):
        self.depth2points(Z)
        dx, dy = np.gradient(Z)
        dx, dy = dx * 5, dy * 5

        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])
        for _ in range(3): colors[:, _] = np_colors

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        # self.pcd.estimate_normals()

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        self.cnt += 1

    def save(self, fpath):
        open3d.io.write_point_cloud(fpath, self.pcd)
