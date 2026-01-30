from typing import Optional, Union, List

import torch

from libraries.NARF.mesh_rendering import render_mesh_, create_mesh, create_mesh_deepcap, create_mesh_deepcap_doublelayer, create_canonmesh_deepcap_doublelayer
from libraries.NARF.pose_utils import transform_pose
from libraries.NeRF.base import NeRFBase
from libraries.NeRF.rendering import render_entire_img


class NARFBase(NeRFBase):
    def __init__(self, z_dim: Union[int, List[int]] = 256, num_bone=1,
                 bone_length=True, parent=None, num_bone_param=None,
                 view_dependent: bool = False):
        super(NARFBase, self).__init__(z_dim, view_dependent)
        self.num_bone = num_bone - 1 if self.origin_location in ["center", "center_fixed"] else num_bone
        self.use_bone_length = bone_length
        #assert parent is not None
        self.parent_id = parent

    def transform_pose(self, pose_to_camera: torch.Tensor, bone_length: torch.Tensor):
        pose_to_camera, bone_length = transform_pose(pose_to_camera, bone_length,
                                                     self.origin_location, self.parent_id)
        return pose_to_camera, bone_length

    def forward0(self, frameidx, rawimg, batchsize, sampled_img_coord, ray_o, ray_d, near, far, mask_at_box, smplposes, smpltrans, smplscale, extrinsics,intrinsic,bone_mask,pose_to_world,pose_to_camera, inv_intrinsics,
                z, z_rend, bone_length,
                render_scale=1, Nc=64, Nf=128, return_intermediate=False,
                truncation_psi=1,
                camera_pose: Optional[torch.Tensor] = None,
                return_disparity=False):
        """
        rendering function for sampled rays
        :param sampled_img_coord: sampled image coordinate
        :param pose_to_camera:
        :param inv_intrinsics:
        :param render_scale:
        :param Nc:
        :param Nf:
        :param return_intermediate:
        :param truncation_psi:
        :param camera_pose:
        :param return_disparity:
        :return: color and mask value for sampled rays
        """
        model_input = {"frameidx": frameidx,"rawimg": rawimg,"ray_o": ray_o, "ray_d": ray_d,"near": near, "far": far, "mask_at_box": mask_at_box,"pose_to_world": pose_to_world,
        "smplposes": smplposes, "smpltrans": smpltrans, "smplscale": smplscale, "extrinsics": extrinsics,"intrinsic": intrinsic,"bone_mask": bone_mask, "z": z, "z_rend": z_rend, "bone_length": bone_length, "truncation_psi": truncation_psi}
        pose_to_camera, model_input["bone_length"] = self.transform_pose(pose_to_camera,
                                                                         model_input["bone_length"])
        return self._forward(sampled_img_coord, pose_to_camera, inv_intrinsics,
                             render_scale, Nc, Nf, return_intermediate,
                             camera_pose, model_input, return_disparity)

    def forward(self, frameidx, rawimg, batchsize, ray_o, ray_d, near, far, mask_at_box, poses, betas, Th, A, joints2D, extrinsics,intrinsic,bone_mask,pose_to_world, inv_intrinsics,
                z, z_rend, bone_length,
                render_scale=1, Nc=64, Nf=128, return_intermediate=False,
                truncation_psi=1,
                camera_pose: Optional[torch.Tensor] = None,
                return_disparity=False):
        """
        rendering function for sampled rays
        :param sampled_img_coord: sampled image coordinate, sampled_img_coord
        :param pose_to_camera:
        :param inv_intrinsics:
        :param render_scale:
        :param Nc:
        :param Nf:
        :param return_intermediate:
        :param truncation_psi:
        :param camera_pose:
        :param return_disparity:
        :return: color and mask value for sampled rays sampled_img_coord, 
        """
        model_input = {"frameidx": frameidx,"rawimg": rawimg,"ray_o": ray_o, "ray_d": ray_d,"near": near, "far": far, "mask_at_box": mask_at_box,"pose_to_world": pose_to_world,
        "poses": poses, "betas": betas, "Th": Th, "A": A, "joints2D": joints2D, "extrinsics": extrinsics,"intrinsic": intrinsic,"bone_mask": bone_mask,"z": z, "z_rend": z_rend, "bone_length": bone_length, "truncation_psi": truncation_psi}
        # pose_to_camera, model_input["bone_length"] = self.transform_pose(pose_to_world,
                                                                         # model_input["bone_length"])#pose_to_camera
        return self._forward(pose_to_world, inv_intrinsics,
                             render_scale, Nc, Nf, return_intermediate,
                             camera_pose, model_input, return_disparity)
                             
    def render_entire_img(self, pose_to_camera, inv_intrinsics, z, z_rend, bone_length, camera_pose=None,
                          render_size=128, Nc=64, Nf=128, semantic_map=False, use_normalized_intrinsics=False,
                          no_grad=True, truncation_psi=1, bbox=None):
        model_input = {"z": z, "z_rend": z_rend, "bone_length": bone_length, "truncation_psi": truncation_psi}
        pose_to_camera, model_input["bone_length"] = self.transform_pose(pose_to_camera,
                                                                         model_input["bone_length"])
        if self.tri_plane_based:
            model_input["tri_plane_feature"] = self.compute_tri_plane_feature(z, bone_length)
        return render_entire_img(self, pose_to_camera, inv_intrinsics, camera_pose,
                                 render_size, Nc, Nf, semantic_map, use_normalized_intrinsics,
                                 no_grad, model_input, bbox=bbox)

    def render_mesh_deepcap(self, frameidx, pose_to_world, z, poses,betas,Th,A,extrinsics,intrinsic, bone_length, voxel_size=0.003,
                    mesh_th=15, truncation_psi=0.4, img_size=128):
        assert z is None or z.shape[0] == 1
        assert bone_length is None or bone_length.shape[0] == 1

        center = pose_to_world[:, 0, :3, 3:].clone()  # (1, 3, 1)
        # model_input = {"z": z, "z_rend": z_rend, "bone_length": bone_length, "truncation_psi": truncation_psi}
        # pose_to_camera, model_input["bone_length"] = self.transform_pose(pose_to_camera,
                                                                         # model_input["bone_length"])
                                                                         
        model_input = {"frameidx": frameidx, "poses": poses, "betas": betas, "Th": Th,"A": A, "extrinsics": extrinsics,"intrinsic": intrinsic,"z": z, "bone_length": bone_length, "truncation_psi": truncation_psi}
        # pose_to_camera, model_input["bone_length"] = self.transform_pose(pose_to_world,
                                                                         # model_input["bone_length"])#pose_to_camera
        
        #if self.tri_plane_based:
            #fix_z = self.latent(torch.LongTensor([0]).to('cuda'))#zself.shapepara.repeat(A.shape[0],1)

            #model_input["tri_plane_feature"] = self.compute_tri_plane_feature(z, bone_length)self.shapepara
            #model_input["tri_plane_feature"], latentws = self.compute_tri_plane_feature_our(fix_z.repeat(A.shape[0],1), truncation_psi)

        # meshes = create_mesh_deepcap(self, pose_to_camera, center=center,
                             # voxel_size=voxel_size,
                             # mesh_th=mesh_th, model_input=model_input)

        # images = render_mesh_(meshes, intrinsic, img_size)#

        # meshes, meshes_cloth = create_mesh_deepcap_doublelayer(self, pose_to_camera, center=center,
                             # voxel_size=voxel_size,
                             # mesh_th=mesh_th, model_input=model_input)self.rawtemplatesmpl

        meshes, meshes_cloth = create_canonmesh_deepcap_doublelayer(self, pose_to_world[:,:, :3, 3], center=center,
                             voxel_size=voxel_size,
                             mesh_th=mesh_th, model_input=model_input)
                             
        #images = render_mesh_(meshes, intrinsic, img_size)#
        images = 0

        #images_cloth = render_mesh_(meshes_cloth, intrinsic, img_size)#
        images_cloth = 0

        return images, meshes, images_cloth, meshes_cloth
