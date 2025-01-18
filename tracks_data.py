import torch
import pypose as pp
from dataclasses import dataclass, field
import einops
import pypose_utils
import itertools
import operator

import rerun as rr
import rerun.blueprint as rrb
import rerun.blueprint.components as rrbc
import argparse

@dataclass
class ClipWithTracks:
    points_2d: torch.Tensor # (N, P, 3) [x,y,is_observed]
    images: torch.Tensor | None = None # (N, 3, H, W)
    points_3d: torch.Tensor | None = None # (N, P, 3) [x,y,z]
    world_from_cam: torch.Tensor | None = None # (N, 6)
    static_mask: torch.Tensor | None = None # (P,)
    intrinsic_mat: torch.Tensor = field(default_factory=lambda: torch.eye(3)) # (3, 3)

if __name__ == "__main__":
    N, P = 45, 40
    w, h, f = 100, 100, 75
    K = torch.tensor([[f,0,w*.5-.5],[0,f,h*.5-.5],[0,0,1]])
    radius = 12.
    times = torch.linspace(0, N, N)
    center = einops.rearrange(torch.stack([torch.cos(0.1*times)*radius,
                                           torch.sin(0.05*times)*radius,
                                           3*radius+0.01*times**2], 
                                           dim=-1), 'n t -> n 1 t')
    points_3d_dynamic = pp.randn_so3(1, P).tensor()
    points_3d_dynamic = center + points_3d_dynamic * points_3d_dynamic.norm(dim=-1, keepdim=True)**-1 * radius * 3

    points_3d_static = torch.stack(
        torch.meshgrid(torch.linspace(-radius*5, radius*5, int(P**.5)),
                       torch.linspace(-radius*5, radius*5, int(P**.5)),
                       torch.tensor(0.), 
                       indexing='ij'
        ),
        dim=-1
    )
    points_3d_static = einops.rearrange(points_3d_static, 'x y z d -> (x y z) d')

    cam_center = einops.rearrange(torch.stack([torch.cos(-0.02*times)*radius*12,
                                               torch.sin(0.02*times)*radius*12,
                                               torch.tensor(radius*5.).broadcast_to(times.shape)], 
                                               dim=-1), 'n t -> n t')

    world_R_cam = rotation=pypose_utils.existing_R_new_yz(new_y_in_existing=torch.tensor([0,1.0,0]).cross(cam_center[0].squeeze(), dim=-1),
                                                          new_z_in_existing=-cam_center[0].squeeze())

    drot = pp.euler2SO3(torch.tensor([.001,-0.015,0]))
    step = drot.repeat(N-1,1) @ pp.randn_SO3(N-1,sigma=0.005)
    world_R_cam = pp.SO3(torch.stack(list(itertools.accumulate(
        step,
        operator.mul,
        initial=world_R_cam))))
    world_from_cam = pypose_utils.create_SE3_from_parts(translation=cam_center,
                                                        rotation=world_R_cam)
    
    pts2d_dynamic = pp.point2pixel(points_3d_dynamic,
                                   intrinsics=K,
                                   extrinsics=world_from_cam.Inv())
    

    parser = argparse.ArgumentParser(description='Show DRR')
    rr.script_add_args(parser)
    args = parser.parse_args()
    
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            name='Tracks',
            origin="world",
            time_ranges=[
                rrb.VisibleTimeRange(
                    timeline="time",
                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-N),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                )
            ],
        ),
    )
    rr.script_setup(
        args,
        "show_pts",
        default_blueprint=blueprint
    )
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis

    rr.log('world/static',
           rr.Points3D(positions=points_3d_static, colors=(128, 128, 128)),
           timeless=True)

    rr.send_columns('world/dynamic', 
                    times=[rr.TimeSequenceColumn("time", times)],
                    components=[
                        rr.Points3D.indicator(),
                        rr.components.Position3DBatch(einops.rearrange(points_3d_dynamic, 
                                                                       'n p d -> (n p) d')).partition([P]*N),
                        rr.components.ColorBatch([(255,255,255)]*N),
                    ],
    )

    rr.send_columns('world/cam/axes/pinhole/pts_dynamic', 
                    times=[rr.TimeSequenceColumn("time", times)],
                    components=[
                        rr.Points2D.indicator(),
                        rr.components.Position2DBatch(einops.rearrange(pts2d_dynamic, 
                                                                       'n p d -> (n p) d')).partition([P]*N),
                        rr.components.ColorBatch([(255,128,255)]*N),
                    ],
    )

        
    rr.log(
        "world/cam",
        [
            rr.Points3D.indicator(),
            rr.components.AxisLength(5.0),
        ], 
        timeless=True
    )
    rr.log(
        "world/cam/axes",
        [
            rr.Points3D.indicator(),
            rr.components.AxisLength(5.0),
        ],
        timeless=True
    )
    rr.log(
        "world/cam/axes/pinhole",
        rr.Pinhole(image_from_camera=K,
                   width=w, height=h,
                   camera_xyz=rr.ViewCoordinates.RDF, 
                   image_plane_distance=radius),
        timeless=True,
    )

    for i, t in enumerate(times):
        pts2d_static = pp.point2pixel(points_3d_static,
                                       intrinsics=K,
                                       extrinsics=world_from_cam[i].Inv()).squeeze()
        
        rr.set_time_sequence('time', t.int())
        rr.log('world/cam',
               rr.Transform3D(mat3x3=world_from_cam[i].rotation().matrix(),
                              translation=world_from_cam[i].translation(),
                              from_parent=False)
               )
        rr.log('world/cam/axes/pinhole/pts_static',
               rr.Points2D(pts2d_static, colors=(0,255,255), radii=-5.),)


    rr.script_teardown(args=args)