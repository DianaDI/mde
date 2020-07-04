import open3d as o3d
from src.data.las2xyz import LasReader

if __name__ == "__main__":
    ## Read .las file
    inFile = LasReader.read("../../data/KirbyLeafOn2017PointCloudEntireSite_split0_part3_52224_8192.las")

    xyz = LasReader.get_scaled_dimensions(inFile)
    inFile.close()
    print(xyz)

    ## Visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[min(xyz[:, 0]), min(xyz[:, 1]), min(xyz[:, 2])])
    o3d.visualization.draw_geometries([pcd, mesh_frame])
