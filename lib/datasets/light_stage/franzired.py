import torch.utils.data as data
# from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
# from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
import trimesh
import torch

from lib.datasets.utils import get_projection_matrix,get_rigid_transformation

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        
        # 注解文件
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        test_view = [i for i in range(num_cams) if i not in cfg.training_view]
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
        self.parents = np.array(parents)

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
        frmstr = os.path.basename(self.ims[index][6:-4]).split('_')
        i = int(frmstr[4])
        frameidx = i + 3550
        msk_path = os.path.join(self.data_root, "training/fg/fg3550" ,str(self.rawcam_inds[index])+'/'+frmstr[0] + '_' + frmstr[1] + '_' + frmstr[2] + '_' + frmstr[
            3] + '_' +str(frameidx) + '.png')     
        msk = imageio.imread(msk_path)[:,:,0]/255

        return msk
        
    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'vertices',
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)
        vert = xyz

        boundthr = 0.2
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

        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        # A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
        A = get_rigid_transformation(poses, joints, parents)
        
        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(3550))
        paramshape = np.load(params_path, allow_pickle=True).item()

        return coord, out_sh, can_bounds, bounds, Rh, Th, vert, A, params['poses'].reshape(-1), paramshape['shapes'].reshape(-1)
  
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

        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(idx))
        smpl_param = np.load(params_path, allow_pickle=True).item()
        Rh = smpl_param['Rh']
        poses = smpl_param['poses'].reshape(-1, 3)
        full_pose = np.concatenate([Rh.reshape(-1, 3), poses.reshape(-1, 72)[:, 3:]], axis=-1)
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
        if frame == cfg.begin_ith_frame + 3550:
            frames = [frame] * 3
        elif frame == cfg.begin_ith_frame + 3551:
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
        frmstr = os.path.basename(self.ims[index][6:-4]).split('_')
        newfrmidx = int(frmstr[4])
        frameidx = newfrmidx+3550
        img_path = 'data/Franzired/FranziRed3550/training/img/img3550/' + str(self.rawcam_inds[index])+'/'+frmstr[0] + '_' + frmstr[1] + '_' + frmstr[2] + '_' + frmstr[
            3]  + '_' +str(frameidx) + '.jpg'
        img = imageio.imread(img_path).astype(np.float32) / 255.
        
        msk = self.get_mask(index)


        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])

        # Pytorch3D 坐标系 (w2c)
        PYTORCH3D_R = np.array(self.cams['R'][cam_ind])
        PYTORCH_T = np.array(self.cams['T'][cam_ind]) / 1000.
        PYTORCH_RT = np.concatenate([PYTORCH3D_R, PYTORCH_T], axis=1).astype(np.float32)
        # OPENCV(COLMAP) 坐标系 (c2w)
        # 因为pytorch3d相机使用的旋转矩阵为c2w,而位移矩阵为w2c,故上述PYTORCH3D的旋转位移矩阵以w2c表示
        # 3DGS 在渲染时使用COLMAP的坐标系，因此需要将PYTORCH3D的坐标系转换为COLMAP坐标系
        c2w = np.linalg.inv(np.concatenate([PYTORCH_RT, [[0.0, 0.0, 0.0, 1.0]]], axis=0))
        opencv_R =  np.stack([c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]], 1)
        # opencv_R =  np.stack([c2w[:3, 0], -c2w[:3, 1], -c2w[:3, 2]], 1)
        opencv_T = c2w[:3, 3:]
        opencv_RT = np.concatenate([opencv_R, opencv_T], axis=1).astype(np.float32)
        opencv_RT = np.concatenate([opencv_RT, [[0.0, 0.0, 0.0, 1.0]]], axis=0).astype(np.float32)
        c2w = opencv_RT

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            #i = int(os.path.basename(img_path)[:-4])
            i = int(os.path.basename(img_path).split('_')[4][:-4])
            frame_index = i

        coord, out_sh, can_bounds, bounds, Rh, Th, vert, A, smplpose, smplshape = self.prepare_input(
            i)
        prior_poses, prior_trans, prior_trans_vel = self.get_prior_smpl_params(i)

        # rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray(
        #     img, msk, K, R, T, can_bounds, self.nrays, self.split)
        
        # FOV
        fovx = 2 * np.arctan(W / (2 * K[0, 0]))
        fovy = 2 * np.arctan(H / (2 * K[1, 1]))
        # 相机投影矩阵和相机位置 3DGS所需
        # TODO: 手动求NEAR和FAR
        zfar = 100.0 # max(zfar, 100.0)
        znear = 0.01 # min(znear, 0.01)
        world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float()
        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).float()
        projection_matrix[..., 2, 0] = - K[0, 2] # DEBUG: 为什么要换？
        projection_matrix[..., 2, 1] = - K[1, 2]
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).numpy()
        camera_center = world_view_transform.inverse()[3, :3]
        cam_intrinsics = K
        
        ret = {
            'img': img,
            'out_sh': out_sh,
            'msk': msk,
            'vert': vert,
            'A': A,
            'smplpose': smplpose,
            'smplshape': smplshape,
            'prior_poses': prior_poses,
            'prior_trans': prior_trans,
            'prior_trans_vel': prior_trans_vel,
            "fovx": fovx,
            "fovy": fovy,
            "world_view_transform": world_view_transform,
            "c2w": c2w,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_intrinsics": cam_intrinsics,
        }
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, 1024, 1024)#cfg.W

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = newfrmidx - cfg.begin_ith_frame
        #latent_index = newfrmidx
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': cam_ind,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        R0 = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R0_snap': R0, 'Th0_snap': Th, 'K': K.astype(np.float32), 'RT': PYTORCH_RT, 'pytorch3d_K': pytorch3d_K}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)


if __name__ == "__main__":
    dataset = Dataset("data/Franzired/FranziRed3550", human="test", ann_file="data/Franzired/FranziRed3550/annots.npy", split="train")
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size = 1)
    import tqdm
    import pyrender
    import smplx
    
    smplmodel = smplx.SMPL("data/SMPL/SMPL_FEMALE.pkl")
    for i in tqdm.tqdm(data_loader):
        # 创建一个简单的三角形网格模型
        beta = i["smplshape"]
        pose = i["prior_poses"][:, -1]
        trans = i["prior_trans"][:, -1]
        smploutput = smplmodel(
            betas = beta,
            body_pose = pose[:, 3:],
            global_orient = pose[:, :3],
            transl = trans,
        )
        
        vertices = smploutput.vertices.detach().numpy()[0]
        
        # vertices = i["vert"][0].numpy()
        faces = np.loadtxt("data/Franzired/FranziRed3550/templatedeformT/smpl/smpltri.txt") - 1
       
        
        c2w = i["c2w"][0].numpy()
        # w2c = i["opengl_RT"][0].numpy()
        # R = w2c[:3, :3]
        # T = w2c[:3, 3]
        # vertices = vertices @ R.T + T
        # c2w[:3, 1:3] *= -1.0
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.random.rand(6890, 3))
        # mesh.show()
        
        img = i["img"][0].numpy()
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        fovy = i["fovy"][0]
        fovx = i["fovx"][0]
        # 创建一个渲染场景
        scene = pyrender.Scene()

        # 创建网格节点
        mesh_node = pyrender.Mesh.from_trimesh(mesh)

        # 添加网格节点到场景中
        scene.add(mesh_node)

        K = i["K"][0].numpy()
        camera_params = {
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, -1],
            'cy': K[1, -1],
            'znear': 0.1,          # 近裁剪面
            'zfar': 100.0          # 远裁剪面
        }
        camera = pyrender.PerspectiveCamera(aspectRatio=W/H, yfov=fovy)
        camera = pyrender.IntrinsicsCamera(**camera_params)
        scene.add(camera, pose=c2w)
        # scene.add(camera)

        # 创建一个渲染器
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

        # 渲染场景
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color[..., :3] * i["msk"].unsqueeze(-1)[0].numpy()
        cv2.imwrite("render_f.png", color)

        # 显示渲染结果
        # pyrender.Viewer(scene, use_raymond_lighting=True)
        break
