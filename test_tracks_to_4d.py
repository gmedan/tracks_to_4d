import einops
import pytest
import torch
from tracks_to_4d import TracksTo4D, TracksTo4DOutputs
import utils
from losses import calculate_costs, TracksTo4DLossMetaParams
from create_tracks_data import create_clip_tracks_data
import pypose as pp

def test_tracks_to_4d_output_shapes():
    N, P = 8, 100  # Number of frames and points
    batch_size = 2
    point2d_measured = torch.randn(batch_size, N, P, 2)  # Random input tensor
    point2d_measured_with_visibility=utils.pad_val_after(point2d_measured, val=1.0)

    num_bases = 12
    model = TracksTo4D(num_bases=num_bases, 
                       d_model=256, 
                       num_heads=16, 
                       num_layers=3, 
                       kernel_size=31)

    outputs = model(point2d_measured_with_visibility)
    
    assert isinstance(outputs, TracksTo4DOutputs)
    assert outputs.bases.shape == (batch_size, P, num_bases, 3)
    assert outputs.gamma.shape == (batch_size, P,)
    assert outputs.cam_from_world_logmap.shape == (batch_size, N, 6)
    assert outputs.coefficients.shape == (batch_size, N, num_bases-1)

    pts3d = outputs.calculate_points()
    assert pts3d.shape == (batch_size, N, P, 3)

    points_3d_in_cameras_coords = outputs.points_3d_in_cameras_coords(points_3d=pts3d)
    assert points_3d_in_cameras_coords.shape == (batch_size, N, P, 3)

    pts2d = outputs.reproject_points(points_3d_in_cameras_coords=points_3d_in_cameras_coords)
    assert pts2d.shape == (batch_size, N, P, 2)

    costs = calculate_costs(predictions=outputs,
                            point2d_measured_with_visibility=point2d_measured_with_visibility)
    
    loss_weights = TracksTo4DLossMetaParams()
    assert costs.calc_loss(loss_weights=loss_weights).shape == torch.Size()


def test_calc_losses_correctness():
    data = create_clip_tracks_data()
    B, N, P, _ = data.points_3d.shape

    first_basis = data.points_3d[:, :1, :, :] # (B, N, P, 3)
    movement_to_frame = data.points_3d[:, 1:, :, :] - first_basis # (B, N, P, 3)
    bases = torch.cat([first_basis, movement_to_frame], dim=1) # (B, K, P, 3) with K = N
    bases = einops.rearrange(bases, 'b k p d -> b p k d') # (B, P, K, 3) with K = N 
    coefficients = einops.rearrange(torch.eye(N)[:,1:], 'n k -> 1 n k') # (B, N, K-1) with K = N 
    
    gamma = torch.tensor([1e9, 1e-5])[data.static_mask.to(torch.int32)]
    cam_from_world_logmap = pp.se3(data.world_from_cam_logmap).Exp().Inv().Log().tensor()
    pred = TracksTo4DOutputs(
        cam_from_world_logmap=cam_from_world_logmap,
        gamma=gamma,
        bases=bases,
        coefficients=coefficients
    )

    pred_pts3d = pred.calculate_points()
    pts3d_diff = data.points_3d - pred_pts3d
    torch.testing.assert_close(pts3d_diff, torch.zeros_like(pts3d_diff))
    
    pred_pts3d_in_cams = pred.points_3d_in_cameras_coords(points_3d=pred_pts3d) # (B, N, P, 3)
    pred_pts2d = pred.reproject_points(points_3d_in_cameras_coords=pred_pts3d_in_cams,
                                       intrinsics=data.intrinsic_mat) # (B, N, P, 2)
    pts2d_diff = data.points_2d - pred_pts2d
    assert pts2d_diff.abs().max().item() < 1e-3

    costs = calculate_costs(predictions=pred,
                            point2d_measured_with_visibility=utils.pad_val_after(data.points_2d, dim=-1, val=1),
                            intrinsics=data.intrinsic_mat)
    
    assert costs.static_cost[:,:,data.static_mask.squeeze(),:].max().item() < -5


if __name__ == "__main__":
    pytest.main()
