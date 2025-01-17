import torch
import torch.nn as nn
import einops
import pypose as pp
from dataclasses import dataclass

from tracks_prediction_data import TracksTo4DOutputs

@dataclass
class TracksTo4DLossWeights:
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

    def calc_loss(self, loss_weights: TracksTo4DLossWeights):
        return loss_weights.reprojection * self.reprojection_loss + \
               loss_weights.static * self.static_loss + \
               loss_weights.in_front * self.in_front_loss + \
               loss_weights.sparse * self.sparse_loss


def calculate_costs(predictions: TracksTo4DOutputs, 
                    point2d_measured_with_visibility: torch.Tensor) -> TracksTo4DCosts:
    visibile = einops.rearrange(point2d_measured_with_visibility[...,-1], 'n p -> n p 1')
    visible_count = visibile.sum()

    pts3d = predictions.calculate_points() # (N, P, 3)
    pts3d_in_cams = predictions.points_3d_in_cameras_coords(points_3d=pts3d) # (N, P, 3)
    pts2d = predictions.reproject_points(points_3d_in_cameras_coords=pts3d_in_cams) # (N, P, 2)
    
    # Eq. 5
    reprojection_errors = predictions.reprojection_errors(
        point2d_predicted=pts2d,
        point2d_measured_with_visibility=point2d_measured_with_visibility) # (N, P, 2)
    reprojection_loss = ((reprojection_errors**2).sum() / visible_count)**.5 
    
    # Eq. 8
    gamma = einops.rearrange(predictions.gamma, 'p -> 1 p 1')
    gamma_inverse = gamma ** -1
    static_cost = torch.log(gamma + gamma**-1 * reprojection_errors**2) * visibile
    static_loss = (static_cost.sum() / visible_count)**.5 

    in_front_cost = -torch.min(torch.tensor(0.0, device=pts3d_in_cams.device), 
                               pts3d_in_cams[..., -1:])
    in_front_loss = (in_front_cost * visibile).sum()
 
    # Eq. 10
    gamma_inverse = gamma_inverse.detach() # detach gamma (1, P, 1)
    sparse_cost = gamma_inverse * einops.reduce(predictions.bases.abs(), 'p k d -> k p 1', 'mean') # (K, P, 1)
    sparse_loss = sparse_cost.mean()

    return TracksTo4DCosts(
        reprojection_errors=reprojection_errors,
        static_cost=static_cost,
        in_front_cost=in_front_cost,
        sparse_cost=sparse_cost,

        reprojection_loss=reprojection_loss,
        static_loss=static_loss,
        in_front_loss=in_front_loss,
        sparse_loss=sparse_loss
    )