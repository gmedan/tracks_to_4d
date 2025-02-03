import itertools
import einops
import pytest
import torch
from tracks_attention import TracksAttention
from positional_encoding import CoordinatePositionalEncoding
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

    pts2d = outputs.reproject_points(points_3d_in_cameras_coords=points_3d_in_cameras_coords,
                                     intrinsics=torch.eye(3))
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


def test_tracks_attention_output_shape():
    batch_size = 2
    n_frame = 8
    n_point = 100
    input_dim = 256
    output_dim = 256

    model = TracksAttention(input_dim=input_dim, 
                            output_dim=output_dim, 
                            num_heads=16, 
                            hidden_layer_dim=2048, 
                            dropout=0.1)
    input_tensor = torch.randn(batch_size, n_frame, n_point, input_dim, 
                               requires_grad=False)
    model.eval()
    output_tensor = model(input_tensor)
    
    assert output_tensor.shape == (batch_size, n_frame, n_point, output_dim)

def coordinate_positional_encoding_reference_impl(x: torch.Tensor, positional_dim: int):
    # to the ref impl expected shape
    x = einops.rearrange(x, 'b n p d -> b p d n')
    # ref impl
    b = torch.tensor([(2 ** j) * torch.pi for j in range(positional_dim)],requires_grad = False)

    x_original_shape=x.shape
    x=x.transpose(2,3)
    x=x.reshape(-1,x.shape[-1])
    proj = torch.einsum('ij, k -> ijk', x, b)  
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  
    pos = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    x=torch.cat((x,pos),dim=1)
    x=x.view(x_original_shape[0],x_original_shape[1],x_original_shape[3],x.shape[-1]).transpose(2,3)
    x=torch.cat((x,torch.ones(x.shape[0],x.shape[1],1,x.shape[3])),dim=2)
    # back to the expected shape
    x = einops.rearrange(x, 'b p d n -> b n p d')
    return x

# permute the possible ways to collapse the dimensions of positional encoding
@pytest.fixture(params=itertools.permutations(['d', 'pos', 'sincos']),
                ids=lambda n: ' '.join(n))
def rearrange_string(request):
    return ' '.join(request.param)

@pytest.fixture
def correct_rearrange_string(request):
    return 'pos sincos d'

def test_coordinate_positional_encoding(rearrange_string: str, 
                                        correct_rearrange_string:str):
    batch_size = 2
    num_frames = 8
    num_points = 100
    positional_dim = 12

    CoordinatePositionalEncoding.REARRANGE_ORDER = rearrange_string
    model = CoordinatePositionalEncoding(positional_dim=positional_dim)

    input_tensor = torch.randn(batch_size, num_frames, num_points, 3)
    output_tensor = model(input_tensor)
    
    expected_output_dim = model.output_dim
    assert output_tensor.shape == (batch_size, num_frames, num_points, expected_output_dim)

    ref_impl_output = coordinate_positional_encoding_reference_impl(input_tensor, positional_dim=positional_dim)
    assert ref_impl_output.shape == output_tensor.shape

    # compare the output of the model with the reference implementation
    # (works with the correct rearrange string only)
    if rearrange_string == correct_rearrange_string:
        torch.testing.assert_close(output_tensor, 
                                   ref_impl_output, 
                                   rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main()
