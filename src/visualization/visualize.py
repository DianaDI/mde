import numpy as np
import open3d as o3d
from src.data.transforms import rebin, minmax_over_nonzero, interpolate_on_missing


def dm2pc(dm):
    ind = np.indices(dm.shape)
    xyz = np.c_[ind[0].flatten(), ind[1].flatten(), dm.flatten()]

    ## Visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[min(xyz[:, 0]), min(xyz[:, 1]), min(xyz[:, 2])])
    o3d.visualization.draw_geometries([pcd, mesh_frame], point_show_normal=True)


def show_dm(path, processed=True, adjust=10):
    dm = np.load(path, allow_pickle=True)
    if not processed:
        dm = dm / 1000
        dm = rebin(dm, (128, 128))
        dm = minmax_over_nonzero(dm)
        mask = (dm >= 0).astype(int)
        dm = dm * mask
        if np.min(mask) == 0:
            dm = interpolate_on_missing(dm)
    dm2pc(dm * adjust)

# show_dm("../../data/KirbyLeafOn2017DMEntireSite_107217_31293.dmp", processed=False)
