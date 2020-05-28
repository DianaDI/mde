from osgeo import gdal
from glob import glob
import numpy as np
from os.path import basename, dirname
import os
from tqdm import tqdm
from src.data.las2xyz import LasReader
import multiprocessing
from joblib import Parallel, delayed

img_width = 512
img_height = 512


def world2pixel(x_world, y_world, xoffset, yoffset, w1, h2):
    px = (x_world - xoffset) / w1
    py = (y_world - yoffset) / h2
    return px-1, py-1


def get_closest_empty_pixel(arr, x, y):
    x = int(round(x))
    y = int(round(y))
    shape = arr.shape
    if arr[x][y] == 0:
        return x, y
    else:
        # loop around a point starting from closest surrounding points
        search_level = 5
        for r in range(search_level):
            step = r + 1
            for i in range(x - step, x + step + 1):
                for j in range(y - step, y + step + 1):
                    if i < shape[0] and j < shape[1]:
                        if arr[i][j] == 0:
                            return i, j
        return None, None


def get_depth_map(path_to_las, path_to_tif):
    depth_map = np.zeros([img_width, img_height])
    xyz = LasReader.get_xyz(path_to_las)
    n = len(xyz)
    raster = gdal.Open(path_to_tif)
    xoffset, w1, h1, yoffset, w2, h2 = raster.GetGeoTransform()
    cnt = 0
    for i in range(n):
        x, y = world2pixel(xyz[i][0], xyz[i][1], xoffset, yoffset, w1, h2)
        x, y = get_closest_empty_pixel(depth_map, x, y)
        if x != None and y != None:
            depth_map[x][y] = round(xyz[i][2], 4)
        else:
            cnt+=1
            print("Empty pixel not found. Depth value was omitted")
    if cnt > 0:
        print(f'Not found correspondences {cnt}/{n}')
    return depth_map


def process_dir(dir):
    print(dir)
    for dir in tqdm(glob(f'{splits_dir}/{dir}')):
        for las in tqdm(glob(f'{dir}/*.las')):
            name = basename(las)
            tif_name_no_fmt = name[:-4].replace("PointCloud", "RGBN")
            dir_tif = basename(dirname(las)).replace("PointCloud", "RGBN")
            if "KirbyLeafOff2017PointCloud" in dir:
                tif_name_no_fmt = tif_name_no_fmt.replace("e_", "ePCCrop_")
                dir_tif = f'{dir_tif}PCCrop'
            corr_tif = f'{splits_dir}/{dir_tif}/{tif_name_no_fmt}.tif'
            map = get_depth_map(las, corr_tif)
            map.dump(f'{save_dir}/{tif_name_no_fmt.replace("RGBN", "DM")}.dmp')


if __name__ == "__main__":
    splits_dir = "/mnt/data/davletshina/datasets/Bera_MDE/splits2/"
    save_dir = f'/mnt/data/davletshina/datasets/Bera_MDE/depth_maps2/'

    try:
        os.makedirs(save_dir)
    except OSError as e:
        print(e)
    las_dirs = ["KirbyLeafOff2017PointCloudEntireSite",
                "KirbyLeafOn2017PointCloudEntireSite"]
    num_cores = multiprocessing.cpu_count()
    print(f'RUNNING ON {num_cores} CPUs')
    Parallel(n_jobs=num_cores)(delayed(process_dir)(d) for d in las_dirs)
