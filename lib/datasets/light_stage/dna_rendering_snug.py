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
import trimesh
from typing import Iterable, Optional, Tuple, Union
import torch
import smplx
from lib.utils.lbs_onlyskining import get_transformed_joints
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        faces_path = os.path.join(data_root, 'templatedeformT/modeltri.txt')
        faces = np.loadtxt(faces_path)-1
        self.faces = faces.astype(np.int32)        
        templateshape_path = os.path.join(data_root, 'templatedeformT/templateshapeT.txt')
        self.templateshape = np.loadtxt(templateshape_path)
        self.mesh = trimesh.Trimesh(self.templateshape, self.faces, process=False)
        
        # self.facevert1 = self.templateshape[self.faces[:,0],:] 
        # #self.facevert1 = facevert1.repeat(8, axis=0)
        # self.facevert2 = self.templateshape[self.faces[:,1],:]
        # #self.facevert2 = facevert2.repeat(8, axis=0)        
        # self.facevert3 = self.templateshape[self.faces[:,2],:] 
        # #self.facevert3 = facevert3.repeat(8, axis=0) 
        
        # faces_path = os.path.join(data_root, 'templatedeformT/cloth/clothes_watertight_face.txt')
        faces_path = os.path.join(data_root, 'templatedeformT/cloth/skirt_w_watertight_faces.txt')
        faces = np.loadtxt(faces_path)-1
        self.clothfaces = faces.astype(np.int32)        
        # templateshape_path = os.path.join(data_root, 'templatedeformT/cloth/clothes_watertight_vert.txt')
        templateshape_path = os.path.join(data_root, 'templatedeformT/cloth/skirt_w_watertight_vert.txt')
        self.clothvert = np.loadtxt(templateshape_path)
        self.clothmesh = trimesh.Trimesh(self.clothvert, self.clothfaces, process=False)
        # faces_path = os.path.join(data_root, 'templatedeformT/cloth/smpl_face.txt')
        faces_path = os.path.join(data_root, 'templatedeformT/desmpl/desmpltri.txt')
        faces = np.loadtxt(faces_path)-1
        self.smplfaces = faces.astype(np.int32)        
        # templateshape_path = os.path.join(data_root, 'templatedeformT/vpersonalshape.txt')
        templateshape_path = os.path.join(data_root, 'templatedeformT/desmpl/desmplvt.txt')
        self.personsmplvert = np.loadtxt(templateshape_path)
        self.smplmesh = trimesh.Trimesh(self.personsmplvert, self.smplfaces, process=False)
        self.smpl_model = smplx.SMPL("tools/data/SMPL/SMPL_FEMALE.pkl")#.to("cuda:0")
        
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        #test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        # NOTE: 训练或者测试的视角
        test_view = [8]#[0,2,4]#[2,8,12]#[1,4,7,10]#[i for i in range(num_cams)]#[2,3,6,8,12,13]#cfg.training_view
        view = cfg.training_view if split == 'train' else test_view
        if len(view) == 0:
            view = [0]

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = (i + cfg.num_train_frame) * i_intv
            ni = cfg.num_novel_pose_frame
            if self.human == 'CoreView_390':
                i = 0
 
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni][::i_intv]
        ]).ravel()

        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni][::i_intv]
        ]).ravel()
        self.rawcam_inds = np.array([
            np.array(cfg.raw_view)[view]
            for ims_data in annots['ims'][i:i + ni][::i_intv]
        ]).ravel()

        self.num_cams = len(view)

        self.nrays = cfg.N_rand

        joints = np.load(os.path.join(self.data_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        parents = np.load(os.path.join(self.data_root, 'parents.npy'), allow_pickle=True)
        #self.parents = {i+1: parents[i] for i in range(0, 23)}
        self.parents = np.array(parents)
        #self.bw = np.load(os.path.join(self.data_root, 'bw.npy'), allow_pickle=True)
        #self.A_all, self.R_all, self.Th_all = self.readsmplpara_allfrm()

    def convert_K_3x3_to_4x4(
            self,
            K: torch.Tensor,
            is_perspective: bool = True) -> torch.Tensor:
        """Convert opencv 3x3 intrinsic matrix to 4x4.

        Args:
            K (Union[torch.Tensor, np.ndarray]):
                Input 3x3 intrinsic matrix, left mm defined.
                [[fx,   0,   px],
                [0,   fy,   py],
                [0,    0,   1]]
            is_perspective (bool, optional): whether is perspective projection.
                Defaults to True.

        Raises:
            TypeError: K is not `Tensor` or `array`.
            ValueError: Shape is not (batch, 3, 3) or (3, 3)

        Returns:
            Union[torch.Tensor, np.ndarray]:
                Output intrinsic matrix.
                for perspective:
                    [[fx,   0,    px,   0],
                    [0,   fy,    py,   0],
                    [0,    0,    0,    1],
                    [0,    0,    1,    0]]

                for orthographics:
                    [[fx,   0,    0,   px],
                    [0,   fy,    0,   py],
                    [0,    0,    1,    0],
                    [0,    0,    0,    1]]
        """
        if isinstance(K, torch.Tensor):
            K = K.clone()
        elif isinstance(K, np.ndarray):
            K = K.copy()

        else:
            raise TypeError('K should be `torch.Tensor` or `numpy.ndarray`, '
                            f'type(K): {type(K)}.')
        if K.shape[-2:] == (4, 4):
            # warnings.warn(
            #     f'shape of K already is {K.shape}, will pass converting.')
            return K
        use_numpy = False
        if K.ndim == 2:
            K = K[None].reshape(-1, 3, 3)
        elif K.ndim == 3:
            K = K.reshape(-1, 3, 3)
        else:
            raise ValueError(f'Wrong ndim of K: {K.ndim}')

        if isinstance(K, np.ndarray):
            use_numpy = True
        if is_perspective:
            if use_numpy:
                K_ = np.zeros((K.shape[0], 4, 4))
            else:
                K_ = torch.zeros(K.shape[0], 4, 4)
            K_[:, :2, :3] = K[:, :2, :3]
            K_[:, 3, 2] = 1
            K_[:, 2, 3] = 1
        else:
            if use_numpy:
                K_ = np.eye(4, 4)[None].repeat(K.shape[0], 0)
            else:
                K_ = torch.eye(4, 4)[None].repeat(K.shape[0], 1, 1)
            K_[:, :2, :2] = K[:, :2, :2]
            K_[:, :2, 3:] = K[:, :2, 2:]
        return K_
    
    def convert_screen_to_ndc(
            self,
            K: Union[torch.Tensor, np.ndarray],
            resolution: Union[int, Tuple[int, int], torch.Tensor, np.ndarray],
            sign: Optional[Iterable[int]] = None,
            is_perspective: bool = True,
            index: int = 0,
            cam_id:int = 0
            ) -> Union[torch.Tensor, np.ndarray]:
        """Convert intrinsic matrix from screen to ndc.

        Args:
            K (Union[torch.Tensor, np.ndarray]): input intrinsic matrix.
            resolution (Union[int, Tuple[int, int], torch.Tensor, np.ndarray]):
                (height, width) of image.
            sign (Optional[Union[Iterable[int]]], optional): xyz axis sign.
                Defaults to None.
            is_perspective (bool, optional): whether is perspective projection.
                Defaults to True.

        Raises:
            TypeError: K should be Tensor or array.
            ValueError: shape of K should be (batch, 4, 4)

        Returns:
            Union[torch.Tensor, np.ndarray]: output intrinsic matrix.
        """
        if sign is None:
            sign = [-1, -1, 1]

        if isinstance(K, torch.Tensor):
            K = K.clone()
        elif isinstance(K, np.ndarray):
            K = K.copy()
        else:
            print(K)
            raise TypeError(
                f'K should be `torch.Tensor` or `np.ndarray`, type(K): {type(K)}, {index}, {cam_id}')
        if K.ndim == 2:
            K = K[None].reshape(-1, 4, 4)
        elif K.ndim == 3:
            K = K.reshape(-1, 4, 4)
        else:
            raise ValueError(f'Wrong ndim of K: {K.ndim}')

        if isinstance(resolution, (int, float)):
            w_src = h_src = resolution
        elif isinstance(resolution, (list, tuple)):
            h_src, w_src = resolution
        elif isinstance(resolution, (torch.Tensor, np.ndarray)):
            resolution = resolution.reshape(-1, 2)
            h_src, w_src = resolution[:, 0], resolution[:, 1]

        aspect_ratio = w_src / h_src
        K[:, 0, 0] /= w_src / 2
        K[:, 1, 1] /= h_src / 2
        if aspect_ratio > 1:
            K[:, 0, 0] *= aspect_ratio
        else:
            K[:, 1, 1] /= aspect_ratio
        if is_perspective:
            K[:, 0, 2] = K[:, 0, 2] / (w_src / 2) - 1
            K[:, 1, 2] = K[:, 1, 2] / (h_src / 2) - 1
            K[:, 0, 2] *= sign[0]
            K[:, 1, 2] *= sign[1]
        else:
            K[:, 0, 3] = K[:, 0, 3] / (w_src / 2) - 1
            K[:, 1, 3] = K[:, 1, 3] / (h_src / 2) - 1
            K[:, 0, 3] *= sign[0]
            K[:, 1, 3] *= sign[1]
        return K
    
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
    
    def get_mask(self, index):
        # msk_path = os.path.join(self.data_root, 'mask_cihp',
        #                         self.ims[index])[:-4] + '.png'
        #msk_path = os.path.join('../rddc_dataset/FranziBlue/training/fg/fg1400', 'mask') + self.ims[index][6:]#self.data_root
        #msk_path = '../rddc_dataset/FranziBlue/training/fg/fg1400' + self.ims[index][6:-4] + '.png'
        # frmstr = os.path.basename(self.ims[index][6:-4]).split('_')
        # i = int(frmstr[4])
        # frameidx = i + 1400
        # msk_path = '../rddc_dataset/FranziBlue/training/fg/fg1400/' + str(self.rawcam_inds[index])+'/'+frmstr[0] + '_' + frmstr[1] + '_' + frmstr[2] + '_' + frmstr[
        #     3] + '_' +str(frameidx) + '.png'
        # msk = imageio.imread(msk_path)[:, :, 0] / 255

        #msk_path = os.path.join(self.data_root, 'mask1') + self.ims[index][6:-4] + '.png'#+ self.ims[index][6:]#self.data_root

        msk_path = os.path.join('tools/data/dnarendering/0080_09_annots', 'mask', self.ims[index])
        mask = imageio.imread(msk_path)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        msk = mask / 255
        #cv2.imwrite('maskt.png', mask)
        # msk_cihp = imageio.imread(msk_path)
        # msk = (msk_cihp != 0).astype(np.uint8)
        #
        # border = 5
        # kernel = np.ones((border, border), np.uint8)
        # msk_erode = cv2.erode(msk.copy(), kernel)
        # msk_dilate = cv2.dilate(msk.copy(), kernel)
        # msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def get_dc(self, index):

        #dc_path = os.path.join(self.data_root, 'dc') + self.ims[index][6:]
        #dc = imageio.imread(dc_path)
        #dc = np.array(dc,dtype=float)
        #dcnormal_path = os.path.join(self.data_root, 'dcnormalnew') + self.ims[index][6:-4] + '.npy'
        #normals = np.load(dcnormal_path).astype(np.float32)
        
        #dx_dt = cv2.Sobel(dc, cv2.CV_16S, 1, 0)
        #dy_dt = cv2.Sobel(dc, cv2.CV_16S, 0, 1)
        #dx_dt = dx_dt[:, :, np.newaxis]
        #dy_dt = dy_dt[:, :, np.newaxis]
        #normals = np.concatenate([dx_dt, dy_dt], axis=2).astype(np.float32)

        #lens = np.sqrt(normals[:, :, 0] ** 2 + normals[:, :, 1] ** 2)
        #eps = 0.00000001
        #lens[lens < eps] = eps
        #normals[:, :, 0] /= lens
        #normals[:, :, 1] /= lens

        #dcpos_path = os.path.join(self.data_root, 'dcpos') + self.ims[index][6:-4] + '.npy'
        dcpos_path = os.path.join('../neuralbody-deformation-occupancy-fixnerf/tools/data/magdalena2000-allviews', 'dcpos') + self.ims[index][6:-4] + '.npy'
        dc = np.load(dcpos_path).astype(np.float32)
        # dc1 =dc.reshape(-1,2).astype(np.long)
        # img1 = np.full([1024, 1024, 3], 255, dtype=np.uint8)
        # img1[dc1[:, 1], dc1[:, 0], :] = 0
        # cv2.imwrite('dc_raw_boudary.png', img1)

        return dc
        
    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'vertices',
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)
        vert = xyz

        boundthr = 0.05
        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:
            min_xyz -= boundthr
            max_xyz += boundthr
        else:
            min_xyz[2] -= boundthr
            max_xyz[2] += boundthr

        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        # params_path = os.path.join(self.data_root, 'smpl_params',
        #                            '{}.npy'.format(i))
        # params = np.load(params_path, allow_pickle=True).item()
        params_path = os.path.join(self.data_root, "smpl_params", "0080_09_smpl_new.npy")
        params = np.load(params_path, allow_pickle=True).item()
        # NOTE: 修改为DNA Rendering的数据 
        # NOTE: Tips:此处的Rh为全局旋转，后面的smpl的poses前三项为0
        Rh = params['global_orient'][i:i+1]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['transl'][i:i+1].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= boundthr
            max_xyz += boundthr
        else:
            min_xyz[2] -= boundthr
            max_xyz[2] += boundthr
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

        poses = params['full_pose'][i].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        # A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
        
        full_pose = np.concatenate([Rh.reshape(-1, 3), poses.reshape(-1, 72)[:, 3:]], axis=-1)
        J_transformed, joint_transforms = get_transformed_joints(torch.from_numpy(params['betas'][i:i+1]), 
                                                                    # sp_input['smplpose'], 
                                                                    torch.from_numpy(full_pose).reshape(-1, 72),
                                                                    self.smpl_model.v_template,
                                                                    self.smpl_model.shapedirs,
                                                                    self.smpl_model.J_regressor,
                                                                    self.smpl_model.parents)
        A = joint_transforms.reshape([-1,24,16])[0].numpy()
        
        # params_path = os.path.join(self.data_root, 'smpl_params',
        #                            '{}.npy'.format(0))
        # paramshape = np.load(params_path, allow_pickle=True).item()

        return coord, out_sh, can_bounds, bounds, Rh, Th, vert, A, params['full_pose'][i].reshape(-1), \
                params['betas'][i].reshape(-1)

    def readsmplpara_allfrm(self):
    
        totalfrmnum = cfg.num_train_frame+cfg.begin_ith_frame
        joints = self.joints
        parents = self.parents
        A_all = []
        R_all = []
        Th_all = []  
        
        for i in range(cfg.begin_ith_frame,totalfrmnum):  
        
            params_path = os.path.join(self.data_root, 'smpl_params','{}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh']
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            Th = params['Th'].astype(np.float32)           
            poses = params['poses'].reshape(-1, 3)
            A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)           
            A_all.append(A[None,...])
            R_all.append(R[None,...])
            Th_all.append(Th[None,...])
        A_all = np.concatenate(A_all)
        R_all = np.concatenate(R_all)
        Th_all = np.concatenate(Th_all)
        
        return A_all, R_all, Th_all
    
    def getsmplpara_neighborfrm(self, i):
        previousfrm = i - 1
        nextfrm = i + 1
        if previousfrm<cfg.begin_ith_frame:
           previousfrm = i
        if nextfrm>=cfg.num_train_frame+cfg.begin_ith_frame:
            nextfrm = i
        previousfrm = previousfrm - cfg.begin_ith_frame
        nextfrm = nextfrm - cfg.begin_ith_frame    
        R_pre = self.R_all[previousfrm,:] 
        Th_pre = self.Th_all[previousfrm,:]  
        A_pre = self.A_all[previousfrm,:]  
        R_next = self.R_all[nextfrm,:]  
        Th_next = self.Th_all[nextfrm,:]  
        A_next = self.A_all[nextfrm,:]  
        
        return R_pre, Th_pre, A_pre, R_next, Th_next, A_next
        
    def readsmplpara_neighborfrm(self, i):
        previousfrm = i - 1
        nextfrm = i + 1
        if previousfrm<cfg.begin_ith_frame:
           previousfrm = i
        if nextfrm>=cfg.num_train_frame+cfg.begin_ith_frame:
            nextfrm = i
            
        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(previousfrm))
        params = np.load(params_path, allow_pickle=True).item()
        
        #with np.load(params_path, allow_pickle=True) as params:
        # with open(params_path, 'rb') as f:
            # params = np.load(f, allow_pickle=True)          
        Rh = params['Rh']
        R_pre = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th_pre = params['Th'].astype(np.float32)
        
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A_pre = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
        
       
        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(nextfrm))
        params = np.load(params_path, allow_pickle=True).item()
        # with open(params_path, 'rb') as f:
            # params = np.load(f, allow_pickle=True)
        #with np.load(params_path, allow_pickle=True) as params:
        # with open(params_path, 'rb') as f:
            # params = np.load(f, allow_pickle=True)        
        Rh = params['Rh']
        R_next = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th_next = params['Th'].astype(np.float32)
        
        poses = params['poses'].reshape(-1, 3)
        A_next = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
            
        return R_pre, Th_pre, A_pre, R_next, Th_next, A_next
        
    def get_sampling_points(self, bounds, N_samples):
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        x_vals = np.random.uniform(0, 1, N_samples)
        y_vals = np.random.uniform(0, 1, N_samples)
        z_vals = np.random.uniform(0, 1, N_samples)
        vals = np.stack([x_vals, y_vals, z_vals], axis=1)
        pts = (max_xyz - min_xyz) * vals + min_xyz
        pts = pts.astype(np.float32)
        return pts
        
    def get_bounds(self, xyz):
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        bounds = bounds.astype(np.float32)
        return bounds

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
        smpl_param = np.load(params_path, allow_pickle=True).item()
        # smpl_param = np.load(os.path.join(self.data_root, f"smpl_params/{idx}.npy"), allow_pickle=True).item()
        # poses = smpl_param["poses"][idx].astype(np.float32)
        poses = smpl_param['full_pose'][idx:idx+1]
        Rh = smpl_param['global_orient'][idx:idx+1]
        full_pose = np.concatenate([Rh.reshape(-1, 3), poses.reshape(-1, 72)[:, 3:]], axis=-1)
        # Rh = smpl_param['Rh']
        # R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        # Th = smpl_param['Th'].astype(np.float32)
        Th = smpl_param['transl'][idx:idx+1].astype(np.float32)
        
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
        if frame == cfg.begin_ith_frame:
            frames = [frame] * 3
        elif frame == cfg.begin_ith_frame + 1:
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
        #img_path = os.path.join('../rddc_dataset/FranziBlue/training/img/img1400', self.ims[index])#self.data_root
        # frmstr = os.path.basename(self.ims[index][6:-4]).split('_')
        # newfrmidx = int(frmstr[4])
        # frameidx = newfrmidx+1400
        # img_path = '../rddc_dataset/FranziBlue/training/img/img1400/' + str(self.rawcam_inds[index])+'/'+frmstr[0] + '_' + frmstr[1] + '_' + frmstr[2] + '_' + frmstr[
        #     3]  + '_' +str(frameidx) + '.jpg'
        # print(img_path)
        # img = imageio.imread(img_path).astype(np.float32) / 255.
        #index = (816-700)*4+1
        img_path = os.path.join('tools/data/dnarendering/0080_09/color', self.ims[index])  # self.data_root
        img = imageio.imread(img_path).astype(np.float32) / 255.
        #print(img_path)
        
        #img = cv2.resize(img, (cfg.W, cfg.H))
        msk = self.get_mask(index)

        #dc = self.get_dc(index)
        cam_ind = self.rawcam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        #D = np.array(self.cams['D'][cam_ind])
        # img = cv2.undistort(img, K, D)
        # msk = cv2.undistort(msk, K, D)

        # R = np.array(self.cams['R'][cam_ind])
        # T = np.array(self.cams['T'][cam_ind]) / 1000.
        # RT = np.concatenate([R, T], axis=1).astype(np.float32)
        # DEBUG: 不转换坐标系
        RT = np.array(self.cams['RT'][cam_ind]).astype(np.float32)
        RT = np.linalg.inv(RT)
        R = RT[:3, :3]
        T = RT[:3, 3:]
        RT = np.concatenate([R, T], axis=1).astype(np.float32)
        # pytorch3d
        pytorch_RT = np.array(self.cams['RT'][cam_ind]).astype(np.float32)
        pytorch_R = pytorch_RT[:3, :3]
        pytorch_T = pytorch_RT[:3, 3:]
        # w2c pytorch3d
        pytorch_R = np.stack([-pytorch_R[:, 0], -pytorch_R[:, 1], pytorch_R[:, 2]], 1)
        new_c2w = np.concatenate([pytorch_R, pytorch_T], 1)
        w2c = np.linalg.inv(np.concatenate((new_c2w, np.array([[0,0,0,1]])), 0))
        pytorch_R, pytorch_T = w2c[:3, :3], w2c[:3, 3:] 
        pytorch_RT = np.concatenate([pytorch_R, pytorch_T], axis=1).astype(np.float32)
        # DEBUG: END
        
        # c2w opencv  
        # DEBUG: 原始座标转换
        # RT = np.array(self.cams['RT'][cam_ind]).astype(np.float32)
        # R = RT[:3, :3]
        # T = RT[:3, 3:]
        # # w2c pytorch3d
        # R = np.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)
        # new_c2w = np.concatenate([R, T], 1)
        # w2c = np.linalg.inv(np.concatenate((new_c2w, np.array([[0,0,0,1]])), 0))
        # R, T = w2c[:3, :3], w2c[:3, 3:] 
        # RT = np.concatenate([R, T], axis=1).astype(np.float32)
        # DEBUG: END
        
        # # NOTE: right mm to left mm
        # r_mat = R.transpose(-1, -2)
        # R *= np.array([-1.0, -1.0, 1.0])
        # t_vec = T
        # t_vec = -np.dot(r_mat, t_vec).reshape((3, 1))
        # RT = np.concatenate([R.transpose(-1, -2), t_vec], axis=1).astype(np.float32)
        # RT = np.concatenate([R.transpose(-1, -2), t_vec], axis=1).astype(np.float32)
        # R = RT[:3, :3]
        # T = RT[:3, 3:]

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        aspect_ratio = float(cfg.W) / float(cfg.H)
        # K[0, 0] *= -1
        # K[1, 1] /= -1*aspect_ratio
        K[:2] = K[:2] * cfg.ratio
        # K = self.convert_K_3x3_to_4x4(K=K)[0]

        # img_path1 = os.path.join(self.data_root, 'img.png')
        # imageio.imwrite(img_path1, img*255)

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        elif self.human in ["0080"]:
            i = int(os.path.basename(img_path).split('.')[0]) - 1
            frame_index = i
        else:
            #i = int(os.path.basename(img_path)[:-4])
            i = int(os.path.basename(img_path).split('_')[4][:-4])
            frame_index = i

        coord, out_sh, can_bounds, bounds, Rh, Th, vert, A, smplpose, smplshape = self.prepare_input(
            i)
        #R_pre, Th_pre, A_pre, R_next, Th_next, A_next = self.getsmplpara_neighborfrm(i)
        
        # rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
        #     img, msk, K, R, T, can_bounds, self.nrays, self.split)

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray(
            img, msk, K, R, T, can_bounds, self.nrays, self.split, frame_index, cam_ind)

        n_sample = 10000
        # FIXME: 用于SDF?
        # meshpts, ind = trimesh.sample.sample_surface_even(self.mesh, n_sample)
        # meshpts = meshpts.astype(np.float32)
        # sdf = np.zeros([n_sample]).astype(np.float32)
        # normal = self.mesh.face_normals[ind].astype(np.float32)

        # meshpts1, ind1 = trimesh.sample.sample_surface_even(self.clothmesh, 5000)
        # meshpts1 = meshpts1.astype(np.float32)
        # sdf1 = np.zeros([5000]).astype(np.float32)
        # normal1 = self.clothmesh.face_normals[ind1].astype(np.float32)
        # meshpts2, ind2 = trimesh.sample.sample_surface_even(self.smplmesh, 5000)
        # meshpts2 = meshpts2.astype(np.float32)
        # sdf2 = np.zeros([5000]).astype(np.float32)
        # normal2 = self.smplmesh.face_normals[ind2].astype(np.float32)
        # meshpts = np.concatenate([meshpts1, meshpts2], axis=0).astype(np.float32)
        # sdf = np.concatenate([sdf1, sdf2], axis=0).astype(np.float32)
        # normal = np.concatenate([normal1, normal2], axis=0).astype(np.float32)
        # FIXME: END
        
        # tbounds = self.get_bounds(self.templateshape)
        # tpose = self.get_sampling_points(tbounds, n_sample // 2)
        # tpose = tpose.astype(np.float32)
        
        
        meshpts1, ind1 = trimesh.sample.sample_surface_even(self.clothmesh, n_sample)
        meshpts_cloth = meshpts1.astype(np.float32)
        sdf_cloth = np.zeros([n_sample]).astype(np.float32)
        normal_cloth = self.clothmesh.face_normals[ind1].astype(np.float32)
        meshpts2, ind2 = trimesh.sample.sample_surface_even(self.smplmesh, n_sample)
        meshpts_smpl = meshpts2.astype(np.float32)
        sdf_smpl = np.zeros([n_sample]).astype(np.float32)
        normal_smpl = self.smplmesh.face_normals[ind2].astype(np.float32)
        
        tbounds = self.get_bounds(self.templateshape)
        tpose = self.get_sampling_points(tbounds, n_sample // 2)
        tpose_smpl = tpose.astype(np.float32)
        tbounds = self.get_bounds(self.clothvert)
        tpose = self.get_sampling_points(tbounds, n_sample // 2)
        tpose_cloth = tpose.astype(np.float32)
        
        # faceweight = np.random.rand(self.faces.shape[0], 3)
        # faceweightsum = np.sum(faceweight,1)
        # faceweight = faceweight/faceweightsum[:,None]
        # meshpts = self.facevert1*faceweight[:,0][:,None]+self.facevert2*faceweight[:,1][:,None]+self.facevert3*faceweight[:,2][:,None]
        # meshpts = meshpts.astype(np.float32)
        # meshpts = meshpts + 0.1 * np.random.rand(self.faces.shape[0], 3)
        # meshpts = meshpts.astype(np.float32)
        
        # NOTE: Snug模拟服装需要至少连续三帧的数据
        prior_poses, prior_trans, prior_trans_vel = self.get_prior_smpl_params(i)
        
        ret = {
            'coord': coord_,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'msk': msk,
            'vert': vert,
            'A': A,
            # 'R_pre': R_pre,
            # 'Th_pre': Th_pre, 
            # 'A_pre': A_pre, 
            # 'R_next': R_next, 
            # 'Th_next': Th_next, 
            # 'A_next': A_next,
            'meshpts_smpl': np.array(meshpts_smpl),
            'sdf_smpl': sdf_smpl,
            'normal_smpl': np.array(normal_smpl),
            'tpose_smpl': tpose_smpl,
            'meshpts_cloth': np.array(meshpts_cloth),
            'sdf_cloth': sdf_cloth,
            'normal_cloth': np.array(normal_cloth),
            'tpose_cloth': tpose_cloth,
            'smplpose': smplpose,
            'smplshape': smplshape,
            # NOTE: Snug所需
            'prior_poses': prior_poses,
            'prior_trans': prior_trans,
            'prior_trans_vel': prior_trans_vel
        }
        pytorch3d_K = K.copy()
        # pytorch3d_K[0, 0] *= -1
        # pytorch3d_K[1, 1] /= -1*aspect_ratio
        # pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(pytorch3d_K, cfg.H * cfg.ratio, cfg.W * cfg.ratio)
        # pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, cfg.H * cfg.ratio, cfg.W * cfg.ratio)
        pytorch3d_K = self.convert_K_3x3_to_4x4(pytorch3d_K)
        pytorch3d_K = self.convert_screen_to_ndc(K=pytorch3d_K, resolution=(cfg.H * cfg.ratio, cfg.W * cfg.ratio), index=index, cam_id=cam_ind)[0]

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = frame_index - cfg.begin_ith_frame
        # latent_index = newfrmidx
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'Rh': Rh,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': cam_ind,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        R0 = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R0_snap': R0, 'Th0_snap': Th, 'K': K.astype(np.float32), 'RT': RT, 'pytorch3d_K': pytorch3d_K,
                "pytorch_RT":pytorch_RT}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
