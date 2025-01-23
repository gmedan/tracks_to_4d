import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input folder for point odyssey.")
    parser.add_argument('input_folder', type=str, nargs='?', default='C:/Users/Guy/Downloads/sample.tar/sample/r4_new_f', help='Path to the input folder')
    rr.script_add_args(parser)

    args = parser.parse_args()

    # info = np.load(Path(args.input_folder) / 'info.npz')
    anno = np.load(Path(args.input_folder) / 'anno.npz')
    N, P, _ = anno['trajs_2d'].shape
    trajs_2d = anno['trajs_2d']
    trajs_3d = anno['trajs_3d']
    visibs = anno['visibs']
    extrinsics = anno['extrinsics']
    intrinsics = anno['intrinsics']
    times = np.arange(N)
    width, height = 960, 540

    
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name='3D',
                origin="world",
                time_ranges=[
                    rrb.VisibleTimeRange(
                        timeline="frame",
                        start=rrb.TimeRangeBoundary.cursor_relative(seq=0),
                        end=rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            ),
            rrb.Spatial2DView(
                name='2D',
                origin="world/cam/pinhole",
                time_ranges=[
                    rrb.VisibleTimeRange(
                        timeline="frame",
                        start=rrb.TimeRangeBoundary.cursor_relative(seq=0),
                        end=rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            ),
        )
    )
    
    rr.script_setup(
        args,
        "show_point_odyssey",
        default_blueprint=blueprint
    )
    rr.log('world/axes', rr.Transform3D(axis_length=1.0), static=True)
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    rr.log('world/cam/pinhole',
            rr.Pinhole(image_from_camera=intrinsics[0,...],
                        image_plane_distance=1.,
                        width=width, height=height,
                        camera_xyz=rr.ViewCoordinates.RDF),
            static=True)
    
    colors= np.array([[128,128,128],[0,255,0]])

    for ii, t in enumerate(times):
        rr.set_time_sequence('frame', t)
        rr.log('world/pts3d',
               rr.Points3D(trajs_3d[ii],
                           colors=colors[visibs[ii].astype(np.int32)]))
        rr.log('world/cam',
               rr.Transform3D(mat3x3=extrinsics[ii,:3,:3],
                              translation=extrinsics[ii,:3,3],
                              from_parent=True,
                              axis_length=1.))
        rr.log('world/cam/pinhole/image',
               rr.ImageEncoded(path=Path(args.input_folder) / 'rgbs' / f'rgb_{ii:05d}.jpg', 
                               format=rr.ImageFormat.JPEG))
        rr.log('world/cam/pinhole/pts2d',
               rr.Points2D(trajs_2d[ii, visibs[ii]]))
        