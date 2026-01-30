import torch.utils.data as data
import numpy as np
import os
import imageio
import cv2
from lib.config import cfg
from plyfile import PlyData
import trimesh
from typing import Iterable, Optional, Tuple, Union
import torch
from lib.datasets.utils import get_projection_matrix,get_rigid_transformation

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        #test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        # NOTE: 训练或者测试的视角
        test_view = [7]#[0,2,4]#[2,8,12]#[1,4,7,10]#[i for i in range(num_cams)]#[2,3,6,8,12,13]#cfg.training_view
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

        msk_path = os.path.join(self.data_root, '0080_09_annots/mask', self.ims[index])
        mask = imageio.imread(msk_path)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        msk = mask / 255

        return msk
        
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
        
        params_path = os.path.join(self.data_root, "smpl_params", "0080_09_smpl_renew.npy")
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
        A = get_rigid_transformation(poses, joints, parents)

        return coord, out_sh, can_bounds, bounds, Rh, Th, vert, A, params['full_pose'][i].reshape(-1), \
                params['betas'][i].reshape(-1)
   
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

        params_path = os.path.join(self.data_root, "smpl_params", "0080_09_smpl_renew.npy")
        smpl_param = np.load(params_path, allow_pickle=True).item()
        poses = smpl_param['full_pose'][idx:idx+1]
        Rh = smpl_param['global_orient'][idx:idx+1]
        full_pose = np.concatenate([Rh.reshape(-1, 3), poses.reshape(-1, 72)[:, 3:]], axis=-1)
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

        img_path = os.path.join(self.data_root, '0080_09/color', self.ims[index])  # self.data_root
        img = imageio.imread(img_path).astype(np.float32) / 255.

        msk = self.get_mask(index)

        #dc = self.get_dc(index)
        cam_ind = self.rawcam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        # DEBUG: 不转换坐标系
        RT = np.array(self.cams['RT'][cam_ind]).astype(np.float32)
        # c2w = np.concatenate([RT, [[0.0, 0.0, 0.0, 1.0]]], axis=0).astype(np.float32)
        c2w = RT
        # DEBUG: OPENGL COORDINATE SYSTEM
        opengl_R =  np.stack([c2w[:3, 0], -c2w[:3, 1], -c2w[:3, 2]], 1)
        opengl_T = c2w[:3, 3:]
        opengl_RT = np.concatenate([opengl_R, opengl_T], axis=1).astype(np.float32)
        opengl_RT = np.concatenate([opengl_RT, np.array([[0,0,0,1]])], axis=0).astype(np.float32)
        # DEBUG: END
        
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

        i = int(os.path.basename(img_path).split('.')[0]) - 1
        frame_index = i

        coord, out_sh, can_bounds, bounds, Rh, Th, vert, A, smplpose, smplshape = self.prepare_input(
            i)
        
        prior_poses, prior_trans, prior_trans_vel = self.get_prior_smpl_params(i)

        # rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray(
        #     img, msk, K, R, T, can_bounds, self.nrays, self.split, frame_index, cam_ind)
        
        # FOV
        fovx = 2 * np.arctan(W / (2 * K[0, 0]))
        fovy = 2 * np.arctan(H / (2 * K[1, 1]))
        # 相机投影矩阵和相机位置 3DGS所需
        # TODO: 手动求NEAR和FAR
        zfar = 100.0 # max(zfar, 100.0)
        znear = 0.01 # min(znear, 0.01)
        world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float()
        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).float()
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
            "opengl_RT": opengl_RT
        }
        pytorch3d_K = K.copy()
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


if __name__ == "__main__":
    from pytorch3d.renderer.cameras import PerspectiveCameras
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        RasterizationSettings, 
        MeshRasterizer,
        BlendParams,
        MeshRenderer,
        SoftSilhouetteShader,
        TexturesVertex
    )
    def create_render(H, W, camera = None, device="cpu",):
        """常见一个渲染器(渲染mesh)

        Args:
            device (str, optional): Defaults to "cpu".
        Returns:
            pytorch3d.renderer: 返回pytorch3d的渲染器
        """
        cameras = camera

        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0,
            faces_per_pixel=1, 
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        # Mesh渲染
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            # shader=SoftPhongShader(
            #     device=device, 
            #     cameras=cameras,
            #     lights=lights
            # )
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        
        return renderer
    
    def create_mesh(vertices, faces, device="cpu"):
        """从顶点和面信息创建pytorch3d的Mesh对象

        Args:
            faces (tensor): 顶点信息 [N, V, 3]
            faces (tensor or numpy.ndarray): 面信息 [N, F, 3]
            device: Defaults to "cpu"
        """
        if not torch.is_tensor(faces):
            faces = torch.from_numpy(faces.astype(np.int32)).to(device)

        if len(faces.shape) == 2:
            faces = torch.tile(faces, [vertices.shape[0], 1, 1])
        verts_rgb = torch.ones_like(vertices) # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        meshes = Meshes(vertices, faces, textures)
        
        return meshes
    
    dataset = Dataset("data/dnarendering/dnarendering", human="test", ann_file="data/dnarendering/dnarendering/annots.npy", split="train")
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size = 1)
    import tqdm
    import pyrender
    import smplx
    
    smplmodel = smplx.SMPL("data/SMPL/SMPL_FEMALE.pkl")
    for num, i in enumerate(tqdm.tqdm(data_loader)):
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
       
        
        c2w = i["opengl_RT"][0].numpy()
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
        # 创建一个透视相机
        camera_params = {
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, -1],
            'cy': K[1, -1],
            'znear': 0.1,          # 近裁剪面
            'zfar': 100.0          # 远裁剪面
        }
        # camera = pyrender.PerspectiveCamera(aspectRatio=float(W)/float(H), yfov=fovy)
        camera = pyrender.IntrinsicsCamera(**camera_params)
        scene.add(camera, pose=c2w)
        # scene.add(camera)

        # 创建一个渲染器
        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

        # 渲染场景
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        # color = color[..., :3] * i["msk"].unsqueeze(-1)[0].numpy()
        color_mask = color[..., :3]*[255, 0, 0] * color[..., -1:]
        gt_msk = i["msk"].unsqueeze(-1)[0].numpy() * 255
        gt_msk = np.repeat(gt_msk, 3, -1)
        merge_msk = cv2.add(color_mask.astype(np.uint8), gt_msk.astype(np.uint8))
        cv2.imwrite(f"render_{num:03d}.png", merge_msk)
        # cv2.imwrite(f"render_{num:03d}.png", color_mask)

        
        # use Pytorch3d to Render
        R, T = torch.split(i['pytorch_RT'], [3, 1], dim=-1)
        R = R.transpose(-1, -2)
        T = T.transpose(-1, -2)
        
        
        cameras = PerspectiveCameras(device='cuda:0',
                                     K=i['pytorch3d_K'].float().to("cuda:0"),
                                     R=R.float().to("cuda:0"),
                                     T=T[0].float().to("cuda:0"))
        render = create_render(H=H, W=W, camera=cameras, device="cuda:0")
        pytorch_mesh = create_mesh(torch.from_numpy(vertices).unsqueeze(0).to("cuda:0"), torch.from_numpy(faces).unsqueeze(0).to("cuda:0"), device="cuda:0")
        predicted_silhouette = render(pytorch_mesh)
        msk = predicted_silhouette.detach().cpu().numpy()[0] * 255
        color_mask = msk[..., :3]*[255, 0, 0] * msk[..., -1:]
        # color = msk[..., :3] * i["msk"].unsqueeze(-1)[0].numpy()
        gt_msk = i["msk"].unsqueeze(-1)[0].numpy() * 255
        gt_msk = np.repeat(gt_msk, 3, -1)
        merge_msk = cv2.add(color_mask.astype(np.uint8), gt_msk.astype(np.uint8))
        cv2.imwrite(f"pytorch3drender_{num:03d}.png", merge_msk)
        # cv2.imwrite(f"pytorch3drender_{num:03d}.png", color_mask)
        # 显示渲染结果
        # pyrender.Viewer(scene, use_raymond_lighting=True)
        if num > 15:
            break