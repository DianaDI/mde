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
    return px, py


def get_closest_empty_pixel(arr, x, y):
    x = int(round(x))
    y = int(round(y))
    shape = arr.shape
    if arr[x][y] == 0:
        return x, y
    else:
        # loop around a point
        search_level = 5
        step = 1
        while step <= search_level:
            for i in range(x - step, x + step + 1):
                for j in range(y - step, y + step + 1):
                    if i < shape[0] and j < shape[1]:
                        if arr[i][j] == 0:
                            return i, j
            step += 1
        return None, None


def get_depth_map(path_to_las, path_to_tif):
    depth_map = np.zeros([img_width, img_height])
    xyz = LasReader.get_xyz(path_to_las)
    n = len(xyz)
    raster = gdal.Open(path_to_tif)
    xoffset, w1, h1, yoffset, w2, h2 = raster.GetGeoTransform()
    for i in range(n):
        x, y = world2pixel(xyz[i][0], xyz[i][1], xoffset, yoffset, w1, h2)
        x, y = get_closest_empty_pixel(depth_map, x, y)
        if x != None and y != None:
            depth_map[x][y] = round(xyz[i][2], 4)
        else:
            print("Empty pixel not found. Depth value was omitted")
    return depth_map


def process_dir(dir):
    print(dir)
    for dir in tqdm(glob(f'{splits_dir}/{dir}')):
        for las in tqdm(glob(f'{dir}/*')):
            name = basename(las)
            name_no_fmt = name[:-4].replace("PointCloud", "RGBN")
            dir_tif = basename(dirname(las)).replace("PointCloud", "RGBN")
            if "KirbyLeafOff2017PointCloud" in dir:
                name_no_fmt = name_no_fmt.replace("e_", "ePCCrop_")
                dir_tif = f'{dir_tif}PCCrop'
            corr_tif = f'{splits_dir}/{dir_tif}/{name_no_fmt}.tif'
            map = get_depth_map(las, corr_tif)
            map.dump(f'{save_dir}/{name_no_fmt.replace("RGBN", "DM")}.dmp')


if __name__ == "__main__":
    splits_dir = "/mnt/data/davletshina/datasets/Bera_MDE/splits/"
    save_dir = f'/mnt/data/davletshina/datasets/Bera_MDE/depth_maps/'

    try:
        os.makedirs(save_dir)
    except OSError as e:
        print(e)
    dirs = ["KirbyLeafOff2017PointCloudEntireSite",
            "KirbyLeafOff2017RGBNEntireSitePCCrop",
            "KirbyLeafOn2017PointCloudEntireSite_split0_part2",
            "KirbyLeafOn2017PointCloudEntireSite_split0_part3",
            "KirbyLeafOn2017PointCloudEntireSite_split0_part4",
            "KirbyLeafOn2017RGBNEntireSite_split0_part2",
            "KirbyLeafOn2017RGBNEntireSite_split0_part3",
            "KirbyLeafOn2017RGBNEntireSite_split0_part4"]
    num_cores = multiprocessing.cpu_count()
    print(f'RUNNING ON {num_cores} CPUs')
    Parallel(n_jobs=num_cores)(delayed(process_dir)(d) for d in dirs)
