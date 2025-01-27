import torch
import einops
from dataclasses import dataclass, field
import pypose as pp

@dataclass
class ClipWithTracks:
    points_2d: torch.Tensor # (N, P, 3) [x,y,is_observed]
    images: torch.Tensor | None = None # (N, 3, H, W)
    points_3d: torch.Tensor | None = None # (N, P, 3) [x,y,z]
    world_from_cam: torch.Tensor | None = None # (N, 6)
    static_mask: torch.Tensor | None = None # (P,)
    intrinsic_mat: torch.Tensor = field(default_factory=lambda: torch.eye(3)) # (3, 3)
    times: torch.Tensor | None = None # (N,)

    @property
    def width(self):
        return self.images.shape[-1]
    
    @property
    def height(self):
        return self.images.shape[-2]
    
    @property
    def num_points(self):
        return self.points_2d.shape[1]

    @property
    def num_frames(self):
        return self.points_2d.shape[0]

@dataclass
class TracksTo4DOutputs:
    """
    Dataclass to hold the outputs of the TRACKSTO4D model forward pass.
    """
    bases: torch.Tensor  # Shape: (B, P, K, 3)
    gamma: torch.Tensor  # Shape: (B, P)
    camera_poses: torch.Tensor  # Shape: (B, N, 6)
    coefficients: torch.Tensor  # Shape: (B, N, K-1)

    def calculate_points(self) -> torch.Tensor:
        """
        Calculate the tensor of 3D points from bases and coefficients.

        The first basis element has an implicit coefficient of 1 and is added separately.

        Returns:
            torch.Tensor: Tensor of points with shape (B, N, P, 3).
        """
        # Ensure the coefficients and bases have compatible shapes
        assert self.coefficients.dim() == 3, "Coefficients must have shape (B, N, K-1)"
        assert self.bases.dim() == 4, "Bases must have shape (B, P, K, 3)"
        assert self.coefficients.shape[2] == self.bases.shape[2] - 1, (
            "Number of coefficients must be K-1 where K is the number of bases"
        )
        
        # Separate the first basis (implicit coefficient = 1) and the remaining bases
        first_basis = self.bases[:, :, 0, :]  # Shape: (B, P, 3)
        remaining_bases = self.bases[:, :, 1:, :]  # Shape: (B, P, K-1, 3)
        
        # Compute the points using the coefficients and remaining bases
        points_from_coefficients = einops.einsum(self.coefficients, remaining_bases,'b n k, b p k m -> b n p m')  # Shape: (N, P, 3)
        
        # Add the first basis to the result using einops.rearrange
        first_basis = einops.rearrange(first_basis, 'b p d -> b 1 p d')
        points = points_from_coefficients + first_basis  # Shape: (B, N, P, 3)
        
        return points

    @property
    def camera_from_world(self):
        return pp.se3(einops.rearrange(self.camera_poses, 'b n s -> b n 1 s')).Exp()
    
    def points_3d_in_cameras_coords(self, points_3d: torch.Tensor):
        return self.camera_from_world.Act(points_3d)
    
    def reproject_points(self, points_3d_in_cameras_coords: torch.Tensor):
        reprojected = pp.point2pixel(
            points=points_3d_in_cameras_coords,
            intrinsics=torch.eye(3, dtype=points_3d_in_cameras_coords.dtype, device=points_3d_in_cameras_coords.device),
        )

        return reprojected
    
