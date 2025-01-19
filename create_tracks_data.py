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
from tracks_data import ClipWithTracks

def create_dynamic_points(N: int, P: int, radius: float) -> torch.Tensor:
    times = torch.linspace(0, N-1, N)
    center = einops.rearrange(torch.stack([torch.cos(0.1*times)*radius,
                                           torch.sin(0.05*times)*radius,
                                           3*radius+0.01*times**2], 
                                           dim=-1), 'n t -> n 1 t')
    pts_3d_dynamic = pp.randn_so3(1, P).tensor()
    pts_3d_dynamic = center + pts_3d_dynamic * pts_3d_dynamic.norm(dim=-1, keepdim=True)**-1 * radius * 3
    return pts_3d_dynamic

def create_static_points(P: int, radius: float, times: torch.Tensor) -> torch.Tensor:
    pts_3d_static = torch.stack(
        torch.meshgrid(torch.linspace(-radius*5, radius*5, int(P**.5)),
                       torch.linspace(-radius*5, radius*5, int(P**.5)),
                       torch.tensor(0.), 
                       indexing='ij'
        ),
        dim=-1
    )
    pts_3d_static = einops.rearrange(pts_3d_static, 'x y z d -> 1 (x y z) d').repeat(len(times), 1, 1)
    return pts_3d_static

def create_camera_trajectory(N: int, radius: float, times: torch.Tensor) -> torch.Tensor:
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
    return world_from_cam

def log_to_rerun(results: ClipWithTracks) -> None:
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis

    n_frames, n_pts = results.points_3d.shape[0], results.points_3d.shape[1]

    colors = torch.tensor([[255,0,255], [255,255,0]])[results.static_mask.int()]
    colors = einops.rearrange(colors, 'p c -> 1 p c').repeat(n_frames, 1, 1)
    
    rr.send_columns('world/pts', 
                    times=[rr.TimeSequenceColumn("time", results.times)],
                    components=[
                        rr.Points3D.indicator(),
                        rr.components.Position3DBatch(einops.rearrange(results.points_3d, 
                                                                       'n p d -> (n p) d')).partition([n_pts] * n_frames),
                        rr.components.ColorBatch(einops.rearrange(colors, 
                                                                  'n p d -> (n p) d')).partition([n_pts] * n_frames),
                        rr.components.RadiusBatch(data=[-3]*n_pts*n_frames).partition([n_pts] * n_frames)
                    ],
    )

    rr.send_columns('world/cam/axes/pinhole/pts', 
                    times=[rr.TimeSequenceColumn("time", results.times)],
                    components=[
                        rr.Points2D.indicator(),
                        rr.components.Position2DBatch(einops.rearrange(results.points_2d, 
                                                                       'n p d -> (n p) d')).partition([n_pts] * n_frames),
                        rr.components.ColorBatch(einops.rearrange(colors, 
                                                                  'n p d -> (n p) d')).partition([n_pts] * n_frames),                    
                        rr.components.RadiusBatch(data=[-1]*n_pts*n_frames).partition([n_pts] * n_frames)                                                                  
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
        rr.Pinhole(image_from_camera=results.intrinsic_mat,
                   height=results.height, width=results.width,
                   camera_xyz=rr.ViewCoordinates.RDF, 
                   image_plane_distance=10.),
        timeless=True,
    )

    for i, t in enumerate(results.times):
        rr.set_time_sequence('time', t.int())
        rr.log('world/cam',
               rr.Transform3D(mat3x3=results.world_from_cam[i].rotation().matrix(),
                              translation=results.world_from_cam[i].translation(),
                              from_parent=False)
               )

if __name__ == "__main__":
    N, P = 45, 40
    w, h, f = 150, 100, 75
    K = torch.tensor([[f,0,w*.5-.5],[0,f,h*.5-.5],[0,0,1]])
    radius = 12.
    
    times = torch.linspace(0, N-1, N)

    pts_3d_dynamic = create_dynamic_points(N, P, radius)
    pts_3d_static = create_static_points(P, radius, times)
    world_from_cam = create_camera_trajectory(N, radius, times)

    pts_3d = torch.cat([pts_3d_dynamic, pts_3d_static], dim=1)
    static_mask = torch.ones(pts_3d.shape[1], dtype=torch.bool)
    static_mask[-pts_3d_static.shape[1]:] = False
    pts_2d = pp.point2pixel(pts_3d, intrinsics=K, extrinsics=world_from_cam.Inv())


    results = ClipWithTracks(
        points_2d=pts_2d,
        points_3d=pts_3d,
        intrinsic_mat=K,
        world_from_cam=world_from_cam,
        images = torch.empty(N,3,h,w),
        static_mask=static_mask,
        times=times
    )

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
                    start=rrb.TimeRangeBoundary.cursor_relative(seq=0),
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

    log_to_rerun(results)

    rr.script_teardown(args=args)