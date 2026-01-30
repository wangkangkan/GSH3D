from typing import Optional

import numpy as np
import torch
from smplx.lbs import (blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform)
from smplx.utils import Tensor

# this code is modified from https://github.com/zju3dv/EasyMocap/blob/master/easymocap/smplmodel/lbs.py
def get_pose(
        smpl,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        pose2rot: bool = True,
        **kwargs
) -> Tensor:
    ''' Forward pass for the SMPL model
        Parameters
        ----------
        global_orient: torch.tensor, optional, shape Bx3
            If given, ignore the member variable and use it as the global
            rotation of the body. Useful if someone wishes to predicts this
            with an external model. (default=None)
        betas: torch.tensor, optional, shape BxN_b
            If given, ignore the member variable `betas` and use it
            instead. For example, it can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        body_pose: torch.tensor, optional, shape Bx(J*3)
            If given, ignore the member variable `body_pose` and use it
            instead. For example, it can used if someone predicts the
            pose of the body joints are predicted from some external model.
            It should be a tensor that contains joint rotations in
            axis-angle format. (default=None)
        transl: torch.tensor, optional, shape Bx3
            If given, ignore the member variable `transl` and use it
            instead. For example, it can used if the translation
            `transl` is predicted from some external model.
            (default=None)
        Returns
        -------
    '''
    # If no shape and pose parameters are passed along, then use the
    # ones from the module
    global_orient = (global_orient if global_orient is not None else
                     smpl.global_orient)
    body_pose = body_pose if body_pose is not None else smpl.body_pose
    betas = betas if betas is not None else smpl.betas

    full_pose = torch.cat([global_orient, body_pose], dim=1)

    batch_size = max(betas.shape[0], global_orient.shape[0],
                     body_pose.shape[0])

    if betas.shape[0] != batch_size:
        num_repeats = int(batch_size / betas.shape[0])
        betas = betas.expand(num_repeats, -1)

    A, T = _get_pose(betas, full_pose, smpl.v_template,
                  smpl.shapedirs,
                  smpl.J_regressor, smpl.parents, smpl.lbs_weights,
                  pose2rot=pose2rot)
                     
    return A, T
    
def get_shape(
        smpl,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        pose2rot: bool = True,
        **kwargs
) -> Tensor:

    # If no shape and pose parameters are passed along, then use the
    # ones from the module
    global_orient = (global_orient if global_orient is not None else
                     smpl.global_orient)
    body_pose = body_pose if body_pose is not None else smpl.body_pose
    betas = betas if betas is not None else smpl.betas

    full_pose = torch.cat([global_orient, body_pose], dim=1)

    batch_size = max(betas.shape[0], global_orient.shape[0],
                     body_pose.shape[0])

    if betas.shape[0] != batch_size:
        num_repeats = int(batch_size / betas.shape[0])
        betas = betas.expand(num_repeats, -1)
                 
    verts, A, T = lbs(betas, full_pose, smpl.v_template, smpl.shapedirs, smpl.posedirs, smpl.J_regressor, smpl.parents,
        smpl.lbs_weights, pose2rot=pose2rot, use_shape_blending=True, use_pose_blending=True)
    
    return verts, A, T
    
def _get_pose(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        lbs_weights: Tensor,
        pose2rot: bool = True,
) -> Tensor:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters
        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        # (N x P) x (P, V * 3) -> N x V x 3
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    
     # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)
        
        
    A[:, :, :3, 3] = J_transformed
    
    return A, T

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, only_shape=False,
        use_shape_blending=True, use_pose_blending=True, J_shaped=None):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    if use_shape_blending:
        v_shaped = v_template + blend_shapes(betas, shapedirs)
        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)
    else:
        v_shaped = v_template.unsqueeze(0).expand(batch_size, -1, -1)
        assert J_shaped is not None
        J = J_shaped[None].expand(batch_size, -1, -1)

    if only_shape:
        return v_shaped, J
    # 3. Add pose blend shapes
    # N x J x 3 x 3
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3)).view([batch_size, -1, 3, 3])#, dtype=torch.float32
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    if use_pose_blending:
        ident = torch.eye(3, dtype=torch.float32, device=device)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=torch.float32)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=torch.float32, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    A[:, :, :3, 3] = J_transformed#save joint position

    return verts, A, T
    
def move_to_origin(bone_pose, scale=0.5):
    """translates the model to the origin"""
    left_hip = 1
    right_hip = 2
    trans = -bone_pose[:, [left_hip, right_hip], :3, 3].mean(axis=1)
    bone_pose = (bone_pose + trans) * scale
    return bone_pose


def axis_transformation(bone_pose: np.ndarray, axis_transformation: np.ndarray = np.array([1, -1, -1])):
    bone_pose[:, :3] *= axis_transformation[None, :, None]
    return bone_pose
