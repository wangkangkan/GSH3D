import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    Textures
)
from pytorch3d.structures import Meshes
from tqdm import tqdm

from libraries.pytorch3d_utils import compute_projection_matrix_from_intrinsics
import trimesh

def render_mesh_(meshes, intrinsics, img_size, render_size=512):
    device = intrinsics.device
    (vertices, triangles, textures) = meshes
    meshes = Meshes(verts=[vertices], faces=[triangles], textures=textures)

    projection_matrix = compute_projection_matrix_from_intrinsics(intrinsics, img_size)
    cameras = FoVPerspectiveCameras(device=device, K=projection_matrix)
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

    raster_settings = RasterizationSettings(
        image_size=render_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(meshes)
    images = images[0, :, :, :3]
    images = (images.cpu().numpy()[::-1, ::-1] * 255).astype("uint8")

    return images


def create_mesh(self, pose_to_camera, center, voxel_size=0.003,
                mesh_th=15, model_input={}):
    import mcubes

    ray_batchsize = self.config.render_bs if hasattr(self.config, "render_bs") else 1048576
    device = pose_to_camera.device
    cube_size = int(1 / voxel_size)

    bins = torch.arange(-cube_size, cube_size + 1) / cube_size
    p = (torch.stack(torch.meshgrid(bins, bins, bins)).reshape(1, 3, -1) + center.cpu()) * self.coordinate_scale

    if self.coordinate_scale != 1:
        pose_to_camera[:, :, :3, 3] *= self.coordinate_scale

    density = []
    for i in tqdm(range(0, p.shape[-1], ray_batchsize)):
        _density = self.calc_density_and_color_from_camera_coord_v2(
            p[:, :, i:i + ray_batchsize].cuda(non_blocking=True),
            pose_to_camera, ray_direction=None,
            model_input=model_input)[0]  # (1, 1, n)
        density.append(_density)
    density = torch.cat(density, dim=-1)
    density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(density, mesh_th)
    vertices = (vertices - cube_size) * voxel_size  # (V, 3)
    vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
    triangles = torch.tensor(triangles.astype("int64")).to(device)

    verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb)
    return (vertices, triangles, textures)

def get_bounds(xyz):
        # vertices_path = os.path.join(self._path[:-6], 'vertices',
                                     # '{}.npy'.format(fname))fname
        # xyz = np.load(vertices_path).astype(np.float32)

        xyz = xyz.float()

        boundthr = 0.3#0.05
        # obtain the original bounds for point sampling
        min_xyz,_ = torch.min(xyz, dim=0)
        max_xyz,_ = torch.max(xyz, dim=0)

        min_xyz -= boundthr
        max_xyz += boundthr

        bounds = torch.stack([min_xyz, max_xyz], axis=0)

        return bounds
        
def create_mesh_deepcap(self, pose_to_camera, center, voxel_size=0.003,
                mesh_th=15, model_input={}):
    import mcubes

    ray_batchsize = self.config.render_bs if hasattr(self.config, "render_bs") else 100000#1048576
    device = pose_to_camera.device
    # cube_size = int(1 / voxel_size)

    # bins = torch.arange(-cube_size, cube_size + 1) / cube_size
    # print(torch.stack(torch.meshgrid(bins, bins, bins)).shape)
    # p = (torch.stack(torch.meshgrid(bins, bins, bins)).reshape(1, 3, -1) + center.cpu()) * self.coordinate_scale
    # p = p.permute(0,2,1)
    
    # if self.coordinate_scale != 1:
        # pose_to_camera[:, :, :3, 3] *= self.coordinate_scale

    can_bounds = get_bounds(pose_to_camera[0, :, :3, 3])
    print(can_bounds)
    voxel_size = [0.005, 0.005, 0.005]#cfg.voxel_size
    x = torch.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                  voxel_size[0])
    y = torch.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                  voxel_size[1])
    z = torch.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                  voxel_size[2])
    p = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), axis=-1)
    print(p.shape)
    cube_size = p.shape[:3]
    p = p.reshape(1, -1, 3)
    
    z = model_input["z"]
    truncation_psi =  model_input["truncation_psi"]

    
    poses, betas = model_input["poses"],model_input["betas"]
    Th = model_input["Th"]
    A = model_input["A"]
    
    extrinsics = model_input["extrinsics"]
    intrinsic = model_input["intrinsic"]
    frameidx = model_input["frameidx"]
    
    bone_length = model_input["bone_length"]
    
    planes = model_input["tri_plane_feature"]
    planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

    density = []
    for i in tqdm(range(0, p.shape[1], ray_batchsize)):
        _density = self.meshdensity_deepcap(frameidx, i, planes, poses.view(-1,24*3), betas, Th, A, extrinsics, intrinsic, p[:, i:i + ray_batchsize, :].cuda(non_blocking=True))
        _density = _density[0]
        # _density = self.calc_density_and_color_from_camera_coord_v2(
            # p[:, :, i:i + ray_batchsize].cuda(non_blocking=True),
            # pose_to_camera, ray_direction=None,
            # model_input=model_input)[0]  # (1, 1, n)
        density.append(_density)
    density = torch.cat(density, dim=0)
    #density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()
    density = density.reshape(cube_size[0],cube_size[1],cube_size[2]).cpu().numpy()
    
    vertices, triangles = mcubes.marching_cubes(-density, mesh_th)
    
    minbounds = can_bounds[0, :][None].repeat(vertices.shape[0], 1)
    #vertices = vertices - np.expand_dims([10, 10, 10], 0).repeat(vertices.shape[0], axis=0)
    vertices = vertices * [0.005, 0.005, 0.005]#cfg.voxel_size
    vertices = torch.tensor(vertices, device=device).float() + minbounds
        
    # vertices = (vertices - cube_size) * voxel_size  # (V, 3)
    # vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
    triangles = torch.tensor(triangles.astype("int64")).to(device)

    verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb)
    return (vertices, triangles, textures)

def create_mesh_deepcap_doublelayer(self, pose_to_camera, center, voxel_size=0.003,
                mesh_th=15, model_input={}):
    import mcubes

    ray_batchsize = self.config.render_bs if hasattr(self.config, "render_bs") else 100000#1048576
    device = pose_to_camera.device
    # cube_size = int(1 / voxel_size)

    # bins = torch.arange(-cube_size, cube_size + 1) / cube_size
    # print(torch.stack(torch.meshgrid(bins, bins, bins)).shape)
    # p = (torch.stack(torch.meshgrid(bins, bins, bins)).reshape(1, 3, -1) + center.cpu()) * self.coordinate_scale
    # p = p.permute(0,2,1)
    
    # if self.coordinate_scale != 1:
        # pose_to_camera[:, :, :3, 3] *= self.coordinate_scale

    can_bounds = get_bounds(pose_to_camera[0, :, :3, 3])
    print(can_bounds)
    voxel_size = [0.005, 0.005, 0.005]#cfg.voxel_size
    x = torch.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                  voxel_size[0])
    y = torch.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                  voxel_size[1])
    z = torch.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                  voxel_size[2])
    p = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), axis=-1)
    print(p.shape)
    cube_size = p.shape[:3]
    p = p.reshape(1, -1, 3)
    
    z = model_input["z"]
    truncation_psi =  model_input["truncation_psi"]

    
    poses, betas = model_input["poses"],model_input["betas"]
    Th = model_input["Th"]
    A = model_input["A"]
    
    extrinsics = model_input["extrinsics"]
    intrinsic = model_input["intrinsic"]
    frameidx = model_input["frameidx"]
    
    bone_length = model_input["bone_length"]
    
    planes = model_input["tri_plane_feature"]
    #planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

    planes_cloth = planes[:,16*3:,:,:]
    planes = planes[:,:16*3,:,:]       
    planes = planes.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
    planes_cloth = planes_cloth.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
    planes = torch.cat([planes,planes_cloth],dim=2)
        
    density_smpl = []
    density_cloth = []
    for i in tqdm(range(0, p.shape[1], ray_batchsize)):
        _density_smpl,_density_cloth = self.meshdensity_deepcap_doublelayer(frameidx, i, planes, poses.view(-1,24*3), betas, Th, A, extrinsics, intrinsic, p[:, i:i + ray_batchsize, :].cuda(non_blocking=True))
        _density_smpl = _density_smpl[0]
        _density_cloth = _density_cloth[0]
        # _density = self.calc_density_and_color_from_camera_coord_v2(
            # p[:, :, i:i + ray_batchsize].cuda(non_blocking=True),
            # pose_to_camera, ray_direction=None,
            # model_input=model_input)[0]  # (1, 1, n)
        density_smpl.append(_density_smpl)
        density_cloth.append(_density_cloth)
    density_smpl = torch.cat(density_smpl, dim=0)
    #density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()
    density_smpl = density_smpl.reshape(cube_size[0],cube_size[1],cube_size[2]).cpu().numpy()
    
    vertices, triangles = mcubes.marching_cubes(-density_smpl, mesh_th)
    
    minbounds = can_bounds[0, :][None].repeat(vertices.shape[0], 1)
    #vertices = vertices - np.expand_dims([10, 10, 10], 0).repeat(vertices.shape[0], axis=0)
    vertices = vertices * [0.005, 0.005, 0.005]#cfg.voxel_size
    vertices = torch.tensor(vertices, device=device).float() + minbounds
        
    # vertices = (vertices - cube_size) * voxel_size  # (V, 3)
    # vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
    triangles = torch.tensor(triangles.astype("int64")).to(device)

    verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb)
    
    density_cloth = torch.cat(density_cloth, dim=0)
    #density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()
    density_cloth = density_cloth.reshape(cube_size[0],cube_size[1],cube_size[2]).cpu().numpy()
    
    vertices_cloth, triangles_cloth = mcubes.marching_cubes(-density_cloth, mesh_th)
    
    minbounds = can_bounds[0, :][None].repeat(vertices_cloth.shape[0], 1)
    #vertices = vertices - np.expand_dims([10, 10, 10], 0).repeat(vertices.shape[0], axis=0)
    vertices_cloth = vertices_cloth * [0.005, 0.005, 0.005]#cfg.voxel_size
    vertices_cloth = torch.tensor(vertices_cloth, device=device).float() + minbounds
        
    # vertices = (vertices - cube_size) * voxel_size  # (V, 3)
    # vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
    triangles_cloth = torch.tensor(triangles_cloth.astype("int64")).to(device)

    verts_rgb_cloth = torch.ones_like(vertices_cloth)[None]  # (1, V, 3)
    textures_cloth = Textures(verts_rgb=verts_rgb_cloth)
    
    return (vertices, triangles, textures), (vertices_cloth, triangles_cloth, textures_cloth)

def create_canonmesh_deepcap_doublelayer(self, canonvert, center, voxel_size=0.003,
                mesh_th=15, model_input={}):
    import mcubes

    ray_batchsize = 100000#self.config.render_bs if hasattr(self.config, "render_bs") else 100000#1048576
    device = canonvert.device
    # cube_size = int(1 / voxel_size)

    # bins = torch.arange(-cube_size, cube_size + 1) / cube_size
    # print(torch.stack(torch.meshgrid(bins, bins, bins)).shape)
    # p = (torch.stack(torch.meshgrid(bins, bins, bins)).reshape(1, 3, -1) + center.cpu()) * self.coordinate_scale
    # p = p.permute(0,2,1)
    
    # if self.coordinate_scale != 1:
        # pose_to_camera[:, :, :3, 3] *= self.coordinate_scale

    can_bounds = get_bounds(canonvert[0])
    print(can_bounds)
    stepsize = 0.008
    voxel_size = [stepsize, stepsize, stepsize]#cfg.voxel_size
    x = torch.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                  voxel_size[0])
    y = torch.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                  voxel_size[1])
    z = torch.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                  voxel_size[2])
    p = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), axis=-1)
    print(p.shape)
    cube_size = p.shape[:3]
    p = p.reshape(1, -1, 3).to(device)
    
    z = model_input["z"]
    truncation_psi =  model_input["truncation_psi"]

    
    poses, betas = model_input["poses"],model_input["betas"]
    Th = model_input["Th"]
    A = model_input["A"]
    
    extrinsics = model_input["extrinsics"]
    intrinsic = model_input["intrinsic"]
    frameidx = model_input["frameidx"]
    
    bone_length = model_input["bone_length"]
    
    planes = 0#model_input["tri_plane_feature"]
    #planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

    # planes_cloth = planes[:,16*3:,:,:]
    # planes = planes[:,:16*3,:,:]       
    # planes = planes.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
    # planes_cloth = planes_cloth.view(len(planes), 3, 16, planes.shape[-2], planes.shape[-1])
    # planes = torch.cat([planes,planes_cloth],dim=2)
   
    density_smpl = []
    density_cloth = []
    #shapeidx = torch.randint(self.allbeta.shape[0],(1,poses.shape[0]))
    smplweight = torch.cuda.FloatTensor(poses.shape[0], 256).normal_()
    #smplweight = torch.rand(poses.shape[0], 256).to('cuda')
    clothweight = torch.rand(poses.shape[0], 2).to('cuda')
    for i in tqdm(range(0, p.shape[1], ray_batchsize)):#cuda(non_blocking=True)
        with torch.no_grad():
            _density_smpl,_density_cloth = self.canonmeshdensity_deepcap_singlelayer_fixgeometry(frameidx, i, smplweight, clothweight,planes, poses.view(-1,24*3), betas, Th, A, extrinsics, intrinsic, p[:, i:i + ray_batchsize, :])
            _density_smpl = _density_smpl[0]
            _density_cloth = _density_cloth[0]
        
        # _density = self.calc_density_and_color_from_camera_coord_v2(
            # p[:, :, i:i + ray_batchsize].cuda(non_blocking=True),
            # pose_to_camera, ray_direction=None,
            # model_input=model_input)[0]  # (1, 1, n)
        density_smpl.append(_density_smpl)
        density_cloth.append(_density_cloth)
    density_smpl = torch.cat(density_smpl, dim=0)
    
    #density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()
    density_smpl = density_smpl.reshape(cube_size[0],cube_size[1],cube_size[2]).cpu().numpy()
    print(mesh_th)
    vertices, triangles = mcubes.marching_cubes(-density_smpl, mesh_th)
    
    minbounds = can_bounds[0, :][None].repeat(vertices.shape[0], 1)
    #vertices = vertices - np.expand_dims([10, 10, 10], 0).repeat(vertices.shape[0], axis=0)
    vertices = vertices * [stepsize, stepsize, stepsize]#cfg.voxel_size
    vertices = torch.tensor(vertices, device=device).float() + minbounds
        
    # vertices = (vertices - cube_size) * voxel_size  # (V, 3)
    # vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
    triangles = torch.tensor(triangles.astype("int64")).to(device)

    verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb)
    
    density_cloth = torch.cat(density_cloth, dim=0)
    #print(density_cloth[density_cloth>0])
    #density = density.reshape(cube_size * 2 + 1, cube_size * 2 + 1, cube_size * 2 + 1).cpu().numpy()
    density_cloth = density_cloth.reshape(cube_size[0],cube_size[1],cube_size[2]).cpu().numpy()
    
    vertices_cloth, triangles_cloth = mcubes.marching_cubes(-density_cloth, mesh_th)
    
    minbounds = can_bounds[0, :][None].repeat(vertices_cloth.shape[0], 1)
    #vertices = vertices - np.expand_dims([10, 10, 10], 0).repeat(vertices.shape[0], axis=0)
    vertices_cloth = vertices_cloth * [stepsize, stepsize, stepsize]#cfg.voxel_size
    vertices_cloth = torch.tensor(vertices_cloth, device=device).float() + minbounds
        
    # vertices = (vertices - cube_size) * voxel_size  # (V, 3)
    # vertices = torch.tensor(vertices, device=device).float() + center[:, :, 0]
    triangles_cloth = torch.tensor(triangles_cloth.astype("int64")).to(device)

    verts_rgb_cloth = torch.ones_like(vertices_cloth)[None]  # (1, V, 3)
    textures_cloth = Textures(verts_rgb=verts_rgb_cloth)
    
    return (vertices, triangles, textures), (vertices_cloth, triangles_cloth, textures_cloth)
    