import torch
import pypose as pp
import torch
from torch import nn
from dataclasses import dataclass
import numpy as np
import typing

from pypose.optim import functional
from pypose.optim import optimizer
from pypose.lietensor import convert
#from pybeyalgo.surgical.models.gimbal import GimbalGeometry


PRIMARY_FROM_ENDUNIT_NOMINAL_METERS = pp.SE3(torch.tensor([0.,0.,400e-3,0.,1.,0.,0.], dtype=torch.float64))
PRIMARY_FROM_ENDUNIT_NOMINAL_MM = pp.SE3(torch.tensor([0.,0.,400,0.,1.,0.,0.], dtype=torch.float64))
METERS_FROM_MM = 1e-3

def rod2SO3(rvec: torch.Tensor, eps=1e-30) -> pp.SO3:
    assert rvec.shape[-1] == 3, 'wrong shape for rvec: must have last dim == 3'
    theta = rvec.norm(dim=-1, keepdim=True)*.5
    rvec_normalized = rvec*(theta*2.0 + eps)**-1
    q = torch.cat([rvec_normalized * torch.sin(theta), torch.cos(theta)], axis=-1)
    return pp.SO3(q)

def create_SE3_from_parts(rotation: pp.SO3, translation: torch.Tensor) -> pp.SE3:
    assert translation.shape[:len(rotation.lshape)] == rotation.lshape, f'translation is wrong shape ({translation.shape}) for concatenating with rotation ({rotation.lshape})'
    return pp.SE3(torch.cat([translation, rotation.tensor()], dim=-1))

def report_pose_error(curr_pose: pp.SE3, gt_pose: pp.SE3):
    _err = (curr_pose.Inv() * gt_pose)
    _err_rot = _err.rotation().Log().norm(dim=-1).mean().item()* (180 / np.pi)
    _err_trans = _err.translation().norm(dim=-1).mean().item()
    print(f"Err Rot (deg): {_err_rot: .9f} | Err Trans (m): {_err_trans: .9f}")

def reflect_points_on_planes(pts: torch.Tensor, planes: torch.Tensor):
    assert pts.shape[-1] == 3 # n x 3
    assert planes.shape[-1] == 3 # m x 3

    pts_dot_planes = torch.einsum('nd, md -> mn', pts, planes)[:,:,None]  # m x n x 1 

    planes_squared = torch.sum(planes**2, dim=-1, keepdim=True)[:,:,None] # m x 1 x 1

    pts = pts[None,:,:]       # 1 x n x 3
    planes = planes[:,None,:] # m x 1 x 3

    return pts - 2.0 * \
        torch.mul(torch.mul(pts_dot_planes - planes_squared, 
                            planes_squared**-1),
                  planes)

# def calc_gimbal_base_from_el_stage_center(
#     az_el_rad: torch.Tensor, 
#     gimbal_geom: GimbalGeometry = GimbalGeometry(az_axis_right_handed_rotation_vector_is_down=True,
#                                                  el_axis_right_handed_rotation_vector_is_right=True)) -> pp.SO3:
#     az_axis = torch.Tensor(gimbal_geom.az_axis_in_right_down_forward_axes)
#     el_axis = torch.Tensor(gimbal_geom.el_axis_in_right_down_forward_axes)
    
#     # Rodrigues(...).inverse() everywhere because Rodrigues provides the rotation operator: v_after_in_cs = Rod(...) * v_before_in_cs, 
#     # but we want the transformation between coordsystems: new_from_old = Rod(...).inverse()

#     az_stage_from_base = rod2SO3(az_axis[None, :] * az_el_rad[...,[0]]).Inv()
    
#     center_el_stage_from_az_stage = rod2SO3(el_axis[None, :] * az_el_rad[...,[1]]).Inv()

#     return (center_el_stage_from_az_stage @ az_stage_from_base).Inv()


class PostInitMeta(type):
    def __call__(cls, *args, **kw):
        instance = super().__call__(*args, **kw)  # < runs __new__ and __init__
        instance.__post_init__()  
        return instance
    
TFORM_SEP = '_T_'

class AutoInverseLieFieldsModule(nn.Module, metaclass=PostInitMeta):
    def __init__(self):
        super().__init__()

    def __post_init__(self):
        tform_prop_keys = [key for key in dir(self) if TFORM_SEP in key]
        
        key_to_inv_tform_keys = {key: TFORM_SEP.join(reversed(key.split(TFORM_SEP))) 
                                for key in tform_prop_keys}
        
        missing = set(key_to_inv_tform_keys.keys()).difference(key_to_inv_tform_keys.values())

        @dataclass
        class Inv:
            field: str

            def __call__(self, instance):
                return getattr(instance, self.field).Inv()

        for key in missing:
            setattr(type(self), key_to_inv_tform_keys[key], property(fget=Inv(key)))

    def dst_from_src(self, dst, src):
        return(getattr(self, f'{dst}_T_{src}'))

    def calc_jacobian(self):
        R = list(self.forward())
        J = functional.modjac(self, vectorize=True, flatten=False)
        params = dict(self.named_parameters())
        params_values = tuple(params.values())
        model = optimizer.RobustModel(self)
        J = [model.flatten_row_jacobian(Jr, params_values) for Jr in J]
        R, _, J = model.normalize_RWJ(R=R, weight=None, J=J)
        return J


    def check_optimizer_jacobian_sensitivity(
                self, 
                optimizer_jacobian: torch.tensor, 
                dof_index_to_dof_name_with_coord_index: typing.List[str],
                num_degenerate_dofs=0, 
                min_singular_value_threshold=1e-4,
                ):
        
        _, d, vh = np.linalg.svd(optimizer_jacobian.cpu().numpy())
        res_dict = {}
        res_dict["singular_values_condition_number"] = np.log10(d[0]/d[-1-num_degenerate_dofs])
        res_dict["first_singular_value"] = d[0]
        res_dict["last_singular_value"] = d[-1-num_degenerate_dofs]
        res_dict["second_last_singular_value"] = d[-2-num_degenerate_dofs]
        res_dict["min_singular_value_threshold"] = min_singular_value_threshold
        least_meaningful_singular_values_indices = np.argwhere(np.bitwise_and(d < min_singular_value_threshold,
                                                                              np.arange(N:=len(d)) < N-num_degenerate_dofs)).ravel()[::-1]
        res_dict["least_meaningful_singular_values_indices"] = least_meaningful_singular_values_indices.tolist()
        res_dict["dof_name_to_nearest_dofs"] = {}
        for singular_idx in least_meaningful_singular_values_indices:
            eigen_vector_highest_abs_values_indices = np.argsort(np.abs(vh[singular_idx]))[::-1][:2]
            res_dict["dof_name_to_nearest_dofs"][dof_index_to_dof_name_with_coord_index[singular_idx]] = [
                dof_index_to_dof_name_with_coord_index[idx]
                for idx in eigen_vector_highest_abs_values_indices.tolist()]
        return res_dict
        
        
    def build_jacobian_sensitivity_dict(self):
        params = dict(self.named_parameters())

        dof_index_to_dof_name_with_coord_index = [f'{name}_{j}'
                                                  for name in params.keys() 
                                                  for j in list(range(getattr(self, name).numel()))]
        
        jac = self.calc_jacobian()
        assert jac.shape[1] == len(dof_index_to_dof_name_with_coord_index)

        num_degenerate_dofs = sum([isinstance(getattr(self, name), pp.LieTensor) and getattr(self, name).ltype in {pp.SE3_type, pp.SO3_type, pp.Sim3_type}
                                   for name in params.keys()])
        optimizer_jacobian_sensitivity_dict = self.check_optimizer_jacobian_sensitivity(jac, 
                                                                                        num_degenerate_dofs=num_degenerate_dofs,
                                                                                        dof_index_to_dof_name_with_coord_index=dof_index_to_dof_name_with_coord_index)
        return optimizer_jacobian_sensitivity_dict

def angle_between_2_vec_numerical_best(x: torch.Tensor,
                                       y: torch.Tensor):
        assert x.shape[-1] == y.shape[-1] == 3, f"x, y must have trailing dimension == 3, but they are: {x.shape=}, {y.shape=}"
        assert len(x.shape) == len(y.shape), f"x, y must have same shape or same length of shape for broadcasting, but they are: {x.shape=}, {y.shape=}"

        x_normalized = x / x.norm(dim=-1, keepdim=True)
        y_normalized = y / y.norm(dim=-1, keepdim=True)
        return torch.arctan2((x_normalized - y_normalized).norm(dim=-1), 
                             (x_normalized + y_normalized).norm(dim=-1)) * 2.0

def existing_R_new_yz(new_y_in_existing: torch.Tensor,
                      new_z_in_existing: torch.Tensor,
                      eps=1e-9) -> pp.SO3:
    assert torch.all(new_y_in_existing.norm(dim=-1) > eps)
    assert torch.all(new_z_in_existing.norm(dim=-1) > eps)
    new_y_in_existing = new_y_in_existing/new_y_in_existing.norm(dim=-1, keepdim=True)
    new_z_in_existing = new_z_in_existing/new_z_in_existing.norm(dim=-1, keepdim=True)
    assert torch.all((new_y_in_existing*new_z_in_existing).sum(dim=-1) < eps), 'not orthogonal'

    new_x_in_existing = torch.linalg.cross(new_y_in_existing, new_z_in_existing, dim=-1)
    existing_R_new = torch.stack([
        new_x_in_existing, 
        new_y_in_existing, 
        new_z_in_existing], dim=1)
    return convert.mat2SO3(existing_R_new)

def existing_T_new_yzp(new_y_in_existing: torch.Tensor,
                       new_z_in_existing: torch.Tensor,
                       new_origin_in_existing: torch.Tensor,
                       eps=1e-9):
    assert new_y_in_existing.shape == new_z_in_existing.shape == new_origin_in_existing.shape
    
    existing_R_new = existing_R_new_yz(new_y_in_existing=new_y_in_existing,
                                       new_z_in_existing=new_z_in_existing,
                                       eps=eps)
    return create_SE3_from_parts(rotation=existing_R_new,
                                 translation=new_origin_in_existing)