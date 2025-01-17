import torch
from einops import reduce, rearrange
from dataclasses import dataclass
import pypose as pp

@dataclass
class TracksTo4DOutputs:
    """
    Dataclass to hold the outputs of the TRACKSTO4D model forward pass.
    """
    bases: torch.Tensor  # Shape: (P, K, 3)
    gamma: torch.Tensor  # Shape: (P,)
    camera_poses: torch.Tensor  # Shape: (N, 6)
    coefficients: torch.Tensor  # Shape: (N, K-1)

    def calculate_points(self) -> torch.Tensor:
        """
        Calculate the tensor of 3D points from bases and coefficients.

        The first basis element has an implicit coefficient of 1 and is added separately.

        Returns:
            torch.Tensor: Tensor of points with shape (N, P, 3).
        """
        # Ensure the coefficients and bases have compatible shapes
        assert self.coefficients.dim() == 2, "Coefficients must have shape (N, K-1)"
        assert self.bases.dim() == 3, "Bases must have shape (P, K, 3)"
        assert self.coefficients.shape[1] == self.bases.shape[1] - 1, (
            "Number of coefficients must be K-1 where K is the number of bases"
        )
        
        # Separate the first basis (implicit coefficient = 1) and the remaining bases
        first_basis = self.bases[:, 0, :]  # Shape: (P, 3)
        remaining_bases = self.bases[:, 1:, :]  # Shape: (P, K-1, 3)
        
        # Compute the points using the coefficients and remaining bases
        points_from_coefficients = torch.einsum('nk,pkm->npm', self.coefficients, remaining_bases)  # Shape: (N, P, 3)
        
        # Add the first basis to the result using einops.rearrange
        first_basis_expanded = rearrange(first_basis, 'p c -> 1 p c')  # Shape: (1, P, 3)
        points = points_from_coefficients + first_basis_expanded  # Shape: (N, P, 3)
        
        return points

    @property
    def camera_from_world(self):
        return pp.se3(self.camera_poses).Exp()
    
    def reproject_points(self, points_3d: torch.Tensor):
        reprojected = pp.point2pixel(
            points=points_3d,
            intrinsics=torch.eye(3, dtype=points_3d.dtype, device=points_3d.device),
            extrinsics=self.camera_from_world
        )

        return reprojected
    
    def reprojection_error(self, 
                           point2d_predicted: torch.Tensor, 
                           point2d_gt_with_visibilty: torch.Tensor):
        
        return point2d_predicted - point2d_gt_with_visibilty[..., :2] * point2d_gt_with_visibilty[..., 2:]
