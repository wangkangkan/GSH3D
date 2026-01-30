import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
import smplx
import torch


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        i = 0
        i = i + cfg.begin_ith_frame
        ni = cfg.num_train_frame
        if cfg.num_render_frame > 0:
            ni = cfg.num_render_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][i:i + ni]
        ])
        self.num_cams = 1

        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[cfg.training_view]
            for ims_data in annots['ims'][i:i + ni]
        ]).ravel()

        self.Ks = np.array(self.cams['K'])[cfg.training_view].astype(
            np.float32)
        self.Rs = np.array(self.cams['R'])[cfg.training_view].astype(
            np.float32)
        self.Ts = np.array(self.cams['T'])[cfg.training_view].astype(
            np.float32) / 1000.
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.ni = ni

        joints = np.load(os.path.join(self.data_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        parents = np.load(os.path.join(self.data_root, 'parents.npy'), allow_pickle=True)
        # self.parents = {i+1: parents[i] for i in range(0, 23)}
        self.parents = np.array(parents)

        # self.bw = np.load(os.path.join(self.data_root, 'bw.npy'), allow_pickle=True)
        
        self.smpl_model = smplx.SMPL("tools/data/SMPL/SMPL_MALE.pkl")

    def prepare_input(self, i):
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = i + 1

        # read xyz, normal, color from the ply file
        # templatesmpl_path = os.path.join(cfg.train_dataset.data_root,
                                         # 'templatedeformT/vpersonalshape.txt')  # smpldeform/vpersonalshape
        # xyz = np.loadtxt(templatesmpl_path)

        vertices_path = os.path.join(self.data_root, 'vertices',
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)
        vert = xyz
        
        smpl_output = self.smpl_model(
            return_verts = True
        )
        xyz = smpl_output.vertices.detach().cpu().numpy()[0]

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if self.human in ['CoreView_362']:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        poses = np.zeros_like(params['poses']).reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(11700))
        paramshape = np.load(params_path, allow_pickle=True).item()
        
        # A = np.zeros_like(A)
        # A[:, :3, :3] = np.eye(3)
        return coord, out_sh, can_bounds, bounds, \
            Rh,Th, vert, A, np.zeros_like(params['poses'].reshape(-1)), paramshape['shapes'].reshape(-1)

    def set_pytorch3d_intrinsic_matrix(self, K, H, W):
        fx = -K[0, 0] * 2.0 / W
        fy = -K[1, 1] * 2.0 / H
        px = -(K[0, 2] - W / 2.0) * 2.0 / W
        py = -(K[1, 2] - H / 2.0) * 2.0 / H
        K = [
            [fx, 0, px, 0],
            [0, fy, py, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
        K = np.array(K)
        return K

    def get_mask(self, i, nv):
        im = self.ims[i, nv]

        #msk_path = os.path.join(self.data_root, 'mask_cihp', im)[:-4] + '.png'
        # msk_path = os.path.join(self.data_root, 'foregroundSegmentation') + im[6:]
        # msk = imageio.imread(msk_path)

        frmstr = os.path.basename(self.ims[i][6:-4]).split('_')
        i = int(frmstr[4])
        frameidx = i + 11700
        msk_path = 'tools/data/Marc11700/training/foregroundSegmentation/' + str(self.cam_inds[i])+'/'+frmstr[0] + '_' + frmstr[1] + '_' + frmstr[2] + '_' + frmstr[3] + '_' +str(frameidx) + '.jpg'

        
        # msk_path = os.path.join(self.data_root, 'foregroundSegmentation') + im[6:]
        mask = imageio.imread(msk_path)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        msk = mask / 255

        # msk_cihp = imageio.imread(msk_path)
        # msk = (msk_cihp != 0).astype(np.uint8)
        #
        # msk = cv2.undistort(msk, self.Ks[nv], self.Ds[nv])
        #
        # border = 5
        # kernel = np.ones((border, border), np.uint8)
        # msk = cv2.dilate(msk.copy(), kernel)

        return msk

    def prepare_inside_pts(self, pts, i):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        # for nv in range(self.ims.shape[1]):
            # ind = inside == 1
            # pts3d_ = pts3d[ind]
        
            # RT = np.concatenate([self.Rs[nv], self.Ts[nv]], axis=1)
            # pts2d = base_utils.project(pts3d_, self.Ks[nv], RT)
        
            # msk = self.get_mask(i, nv)
            # H, W = msk.shape
            # pts2d = np.round(pts2d).astype(np.int32)
            # pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            # pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            # msk_ = msk[pts2d[:, 1], pts2d[:, 0]]
        
            # inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    def get_smpl_params(self, idx):
        """获取SMPL参数

        Args:
            idx (int): 想要获取的SMPL参数的帧数

        Returns:
            poses: SMPL pose参数
            R: SMPL世界座标旋转矩阵
            Th: SMPL世界座标位移
        """
        params_path = os.path.join(self.data_root, "smpl_params", "0080_09_smpl_new.npy")
        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(idx))
        smpl_param = np.load(params_path, allow_pickle=True).item()
        # smpl_param = np.load(os.path.join(self.data_root, f"smpl_params/{idx}.npy"), allow_pickle=True).item()
        # poses = smpl_param["poses"][idx].astype(np.float32)
        # poses = smpl_param['full_pose'][idx:idx+1]
        # Rh = smpl_param['global_orient'][idx:idx+1]
        Rh = smpl_param['Rh']
        poses = smpl_param['poses'].reshape(-1, 3)
        full_pose = np.concatenate([Rh.reshape(-1, 3), poses.reshape(-1, 72)[:, 3:]], axis=-1)
        # Rh = smpl_param['Rh']
        # R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        # Th = smpl_param['Th'].astype(np.float32)
        Th = smpl_param['Th'].reshape(-1, 3).astype(np.float32)
        
        return full_pose, Th
    
    def get_prior_smpl_params(self, frame):
        """获取共计三帧的SMPL参数，用于服装变形网络

        Args:
            frame (int): SMPL参数的帧数

        Returns:
            prior_poses: 包含对应帧前两帧的SMPL参数
            prior_trans: 包含对应帧前两帧的SMPL世界坐标旋转矩阵
            prior_trans_vel: 包含对应帧前两帧的SMPL世界坐标的位移
        """
        prior_poses = []
        prior_trans = []
        if frame == cfg.begin_ith_frame + 11699:
            frames = [frame] * 3
        elif frame == cfg.begin_ith_frame + 11700:
            frames = [frame] * 3
        else:
            frames = [frame-2, frame-1, frame] 
        
        for f in frames:
            poses, T = self.get_smpl_params(f)
            prior_poses.append(poses)
            prior_trans.append(T)
            
        prior_poses = np.concatenate(prior_poses, axis=0)
        prior_trans = np.concatenate(prior_trans, axis=0)
        prior_trans_vel = self.finite_diff(prior_trans, 1 / 30)
        
        return prior_poses, prior_trans, prior_trans_vel

    def finite_diff(self, x, h, diff=1):
        """计算速度

        Args:
            x : 顶点位置
            h : 时间间隔
            diff (int, optional): 需要求微分的阶数. Defaults to 1.

        Returns:
            _type_: _description_
        """
        if diff == 0:
            return x

        v = np.zeros(x.shape, dtype=x.dtype)
        v[1:] = (x[1:] - x[0:-1]) / h

        return self.finite_diff(v, h, diff-1)  
 

    def __getitem__(self, index):

        #index = 984 - 850#700

        i = index
        latent_index = index
        # frame_index = index + cfg.begin_ith_frame
        frame_index = index + 11700

        # msk = self.get_mask(index,0)

        coord, out_sh, can_bounds, bounds, Rh, Th, vert, A, smplpose, smplshape = self.prepare_input(
            frame_index)
        
        prior_poses, prior_trans, prior_trans_vel = self.get_prior_smpl_params(frame_index)

        voxel_size = [0.005, 0.005, 0.005]#cfg.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, i)

        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'pts': pts,
            'inside': inside,
            'vert': vert,
            'A': A,
            'msk': vert,
            # 'meshpts_smpl': msk,
            # 'sdf_smpl': msk,
            # 'normal_smpl': msk,
            # 'tpose_smpl': msk,
            # 'meshpts_cloth': msk,
            # 'sdf_cloth': msk,
            # 'normal_cloth': msk,
            # 'tpose_cloth': msk,
            'meshpts_smpl': vert,
            'sdf_smpl': vert,
            'normal_smpl': vert,
            'tpose_smpl': vert,
            'meshpts_cloth_up': vert,
            'sdf_cloth_up': vert,
            'normal_cloth_up': vert,
            'tpose_cloth_up': vert,
            'meshpts_cloth_low': vert,
            'sdf_cloth_low': vert,
            'normal_cloth_low': vert,
            'tpose_cloth_low': vert,
            'smplpose': smplpose,
            'smplshape': smplshape,
            'prior_poses': prior_poses,
            'prior_trans': prior_trans,
            'prior_trans_vel': prior_trans_vel
        }
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.
        RT = np.concatenate([R, T], axis=1).astype(np.float32)
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, cfg.H, cfg.W)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'bounds': bounds,
            'can_bounds': can_bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'RT': RT,
            'K': K.astype(np.float32),
            'pytorch3d_K': pytorch3d_K
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.ni
