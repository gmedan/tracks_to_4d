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

from tracks_to_4d import TracksTo4D, TracksTo4DOutputs
from losses import TracksTo4DLossMetaParams, calculate_costs, calculate_pretrain_loss
from utils import pad_val_after

def create_dynamic_points(N: int, P: int, radius: float) -> torch.Tensor:
    times = torch.linspace(0, N-1, N)
    center = einops.rearrange(torch.stack([torch.cos(0.1*times)*radius,
                                           torch.sin(0.05*times)*radius,
                                           3*radius+0.01*times**2], 
                                           dim=-1), 'n t -> 1 n 1 t')
    pts_3d_dynamic = pp.randn_so3(1, 1, P).tensor()
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
    pts_3d_static = einops.repeat(
        einops.rearrange(pts_3d_static, 'x y z d -> 1 1 (x y z) d'),
        '1 1 p d -> 1 n p d', 
        n=len(times))
    return pts_3d_static

def create_camera_trajectory(N: int, radius: float, times: torch.Tensor) -> torch.Tensor:
    cam_center = einops.rearrange(torch.stack([torch.cos(-0.02*times)*radius*12,
                                               torch.sin(0.02*times)*radius*12,
                                               torch.tensor(radius*5.).broadcast_to(times.shape)], 
                                               dim=-1), 'n t -> n t')

    world_R_cam = pypose_utils.existing_R_new_yz(new_y_in_existing=torch.tensor([0,1.0,0]).cross(cam_center[0].squeeze(), dim=-1),
                                                 new_z_in_existing=-cam_center[0].squeeze())

    drot = pp.euler2SO3(torch.tensor([.001,-0.015,0]))
    step = drot.repeat(N-1,1) @ pp.randn_SO3(N-1,sigma=0.005)
    world_R_cam = pp.SO3(torch.stack(list(itertools.accumulate(
        step,
        operator.mul,
        initial=world_R_cam))))
    world_from_cam = pypose_utils.create_SE3_from_parts(translation=cam_center,
                                                        rotation=world_R_cam)
    return einops.rearrange(world_from_cam, 'n d -> 1 n d')

def log_to_rerun(data: ClipWithTracks) -> None:
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis

    _, n_frames, n_pts, _ = data.points_3d.shape

    colors = torch.tensor([[255,0,255], [255,255,0]])[data.static_mask.int()]
    colors = einops.repeat(
        colors,
        '1 p c -> n p c',
        n=n_frames)
    
    rr.send_columns('world/pts', 
                    times=[rr.TimeSequenceColumn("time", data.times[0])],
                    components=[
                        rr.Points3D.indicator(),
                        rr.components.Position3DBatch(einops.rearrange(data.points_3d, 
                                                                       'b n p d -> (b n p) d')).partition([n_pts] * n_frames),
                        rr.components.ColorBatch(einops.rearrange(colors, 
                                                                    'n p c -> (n p) c')).partition([n_pts] * n_frames),
                        rr.components.RadiusBatch(data=[-3]*n_pts*n_frames).partition([n_pts] * n_frames)
                    ],
    )

    rr.send_columns('world/cam/axes/pinhole/pts', 
                    times=[rr.TimeSequenceColumn("time", data.times[0])],
                    components=[
                        rr.Points2D.indicator(),
                        rr.components.Position2DBatch(einops.rearrange(data.points_2d, 
                                                                       'b n p d -> (b n p) d')).partition([n_pts] * n_frames),
                        rr.components.ColorBatch(einops.rearrange(colors, 
                                                                  'n p c -> (n p) c')).partition([n_pts] * n_frames),                    
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
        rr.Pinhole(image_from_camera=data.intrinsic_mat[0],
                   height=data.height, width=data.width,
                   camera_xyz=rr.ViewCoordinates.RDF, 
                   image_plane_distance=10.),
        timeless=True,
    )

    for i, t in enumerate(data.times[0]):
        rr.set_time_sequence('time', t.int())
        world_from_cam = pp.se3(data.world_from_cam_logmap[0,i]).Exp()
        rr.log('world/cam',
               rr.Transform3D(mat3x3=world_from_cam.rotation().matrix(),
                              translation=world_from_cam.translation(),
                              from_parent=False)
               )

def create_clip_tracks_data() -> ClipWithTracks:
    N, P = 45, 40
    w, h, f = 150, 100, 75
    K = torch.tensor([[f,0,w*.5-.5],[0,f,h*.5-.5],[0,0,1]])
    radius = 12.
    
    times = torch.linspace(0, N-1, N)

    pts_3d_dynamic = create_dynamic_points(N, P, radius)
    pts_3d_static = create_static_points(P, radius, times)
    world_from_cam = create_camera_trajectory(N, radius, times)

    pts_3d = torch.cat([pts_3d_dynamic, pts_3d_static], dim=-2)
    static_mask = torch.zeros(1, pts_3d.shape[-2], dtype=torch.bool)
    static_mask[..., -pts_3d_static.shape[-2]:] = True
    pts_2d = pp.point2pixel(pts_3d, intrinsics=K, extrinsics=world_from_cam.Inv())


    data = ClipWithTracks(
        points_2d=pts_2d,
        points_3d=pts_3d,
        intrinsic_mat=einops.rearrange(K, 'd1 d2 -> 1 d1 d2'),
        world_from_cam_logmap=world_from_cam.Log(),
        images = torch.empty(1,N,3,h,w),
        static_mask=static_mask,
        times=einops.rearrange(times, 'n -> 1 n')
    )
    
    return data

if __name__ == "__main__":

    data = create_clip_tracks_data()

    parser = argparse.ArgumentParser(description='Tracks to 4D')
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

    log_to_rerun(data)

    model = TracksTo4D(num_bases=2).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_metaparams = TracksTo4DLossMetaParams()

    point2d_with_visibility = pad_val_after(data.points_2d, dim=-1, val=1).double()

    bases_colors = (torch.rand(model.num_bases, 3)*255).int()
    bases_colors = einops.repeat(bases_colors, 
                                 'k d -> (p k) d', p=data.num_points, d=3)
    pretrain_epochs = 50
    target_world_from_cam = pp.se3(data.world_from_cam_logmap[0,0].double()).Exp()

    rr.set_time_sequence('time', data.times[0,0].int())
    for epoch in range(3000):
        optimizer.zero_grad()
        pred: TracksTo4DOutputs = model(point2d_with_visibility)
        loss = calculate_pretrain_loss(predictions=pred, 
                                       target_world_from_cam=target_world_from_cam) \
               if epoch < pretrain_epochs else \
               calculate_costs(predictions=pred, 
                               point2d_measured_with_visibility=point2d_with_visibility).calc_loss(loss_weights=loss_metaparams)
        
        loss.retain_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")

        
        rr.log('world/pred/bases', 
               rr.Points3D(einops.rearrange(pred.bases.detach(), '1 p k d -> (p k) d', d=3),
                           colors=bases_colors),
               timeless=True
        )
        world_from_cameras = pred.camera_from_world.Inv().detach().squeeze() # (N, 7)
        for i, wTc in enumerate(world_from_cameras):
            rr.log(f'world/pred/cams/{i}', 
                   rr.Transform3D(mat3x3=wTc.rotation().matrix(),
                                  translation=wTc.translation(),
                                  from_parent=False),
                   rr.Pinhole(image_from_camera=data.intrinsic_mat,
                              height=data.height, width=data.width,
                              camera_xyz=rr.ViewCoordinates.RDF, 
                              image_plane_distance=10.),
            )

    rr.script_teardown(args=args)