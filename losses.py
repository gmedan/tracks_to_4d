import torch
import torch.nn as nn
import einops
import pypose as pp
from dataclasses import dataclass

from tracks_data import TracksTo4DOutputs

@dataclass
class TracksTo4DLossMetaParams:
    reprojection: float = 50.
    static: float = 1.
    in_front: float = 1.
    sparse: float = 1e-3

@dataclass
class TracksTo4DCosts:
    reprojection_errors: torch.Tensor
    static_cost: torch.Tensor
    in_front_cost: torch.Tensor
    sparse_cost: torch.Tensor

    reprojection_loss: torch.Tensor
    static_loss: torch.Tensor
    in_front_loss: torch.Tensor
    sparse_loss: torch.Tensor

    def calc_loss(self, loss_weights: TracksTo4DLossMetaParams):
        return loss_weights.reprojection * self.reprojection_loss + \
               loss_weights.static * self.static_loss + \
               loss_weights.in_front * self.in_front_loss + \
               loss_weights.sparse * self.sparse_loss


def calculate_costs(predictions: TracksTo4DOutputs, 
                    point2d_measured_with_visibility: torch.Tensor) -> TracksTo4DCosts:
    visible = einops.rearrange(point2d_measured_with_visibility[...,-1], 'n p -> n p 1')
    visible_or_nan = visible/visible
    visible_count = visible.sum()

    pts3d = predictions.calculate_points() # (N, P, 3)
    pts3d_in_cams = predictions.points_3d_in_cameras_coords(points_3d=pts3d) # (N, P, 3)
    pts2d = predictions.reproject_points(points_3d_in_cameras_coords=pts3d_in_cams) # (N, P, 2)
    
    first_basis_3d_in_cams = predictions.points_3d_in_cameras_coords(
        points_3d=einops.rearrange(predictions.bases[:, 0:1, :], 
                                   'p 1 d -> 1 p d')) # (N, P, 3)
    first_basis_static_approximation_2d = predictions.reproject_points(points_3d_in_cameras_coords=first_basis_3d_in_cams) # (N, P, 2)
    
    # Eq. 5
    reprojection_errors = pts2d - point2d_measured_with_visibility[..., :2] # (N, P, 2)
    reprojection_errors = reprojection_errors * visible
    reprojection_loss = ((reprojection_errors**2).sum() / visible_count)**.5 

    # Eq. 8    
    first_basis_reprojection_errors = first_basis_static_approximation_2d - \
                                      point2d_measured_with_visibility[..., :2] # (N, P, 2)
    gamma = einops.rearrange(predictions.gamma, 'p -> 1 p 1').abs() # can't use negative gammas in eq 8
    gamma_inverse = gamma ** -1
    static_cost = torch.log(gamma + gamma**-1 * first_basis_reprojection_errors**2) * visible
    static_loss = (static_cost.sum() / visible_count)**.5 

    # Eq. 9
    in_front_cost = -torch.min(torch.tensor(0.0, device=pts3d_in_cams.device), 
                               pts3d_in_cams[..., -1:])
    in_front_loss = (in_front_cost * visible).sum() / visible_count
 
    # Eq. 10
    gamma_inverse = gamma_inverse.detach() # detach gamma (1, P, 1)
    sparse_cost = gamma_inverse * einops.reduce(predictions.bases[:,1:,:].abs(), # exclude static first base
                                                'p k d -> k p 1', 'mean') # (K-1, P, 1)
    sparse_loss = sparse_cost.mean()

    return TracksTo4DCosts(
        reprojection_errors=reprojection_errors*visible_or_nan,
        static_cost=static_cost*visible_or_nan,
        in_front_cost=in_front_cost*visible_or_nan,
        sparse_cost=sparse_cost,

        reprojection_loss=reprojection_loss,
        static_loss=static_loss,
        in_front_loss=in_front_loss,
        sparse_loss=sparse_loss
    )