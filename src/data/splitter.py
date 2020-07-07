"""
Parallel sliding window splitter of large .tif and according .las files
To split las2las and gdalwarp console tools are used.
"""
from osgeo import gdal
import os
import subprocess
from subprocess import DEVNULL, STDOUT
from tqdm import tqdm
from os.path import basename
import multiprocessing
from joblib import Parallel, delayed
from src.data import IMG_WIDTH

log = True


def exec_console_cmd(cmd):
    try:
        if log:
            res = subprocess.call(cmd, shell=True)
        else:
            res = subprocess.call(cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except OSError as e:
        print(e)


def crop_pc_by_bounds(path, out, xmin, ymin, xmax, ymax):
    cmd = f'las2las {path} --output {out} --minx {xmin} --miny {ymin} --maxx {xmax} --maxy {ymax}'
    if log: print("\nCROPPING PC...")
    exec_console_cmd(cmd)


def crop_tif_by_bounds(path, out, xmin, ymin, xmax, ymax):
    cmd = f'gdalwarp -te {xmin} {ymin} {xmax} {ymax} {path} {out}'
    if log: print("\nCROPPING TIF...")
    exec_console_cmd(cmd)


def pixel2world_bounds(xmin, ymin, xmax, ymax):
    posXmin = xoffset + w1 * xmin + h1 * ymin
    posYmin = yoffset + w2 * xmin + h2 * ymin
    posX = xoffset + w1 * xmax + h1 * ymax
    posY = yoffset + w2 * xmax + h2 * ymax
    return posXmin, posYmin, posX, posY


def process_row(y):
    x = window - 1
    xmin = 0
    ymin = y - step + 1
    while x < width:
        posXmin, posYmin, posX, posY = pixel2world_bounds(xmin, ymin, x, y)
        out_pc_path = pc_out_dir + f'/{pc_name}_{ymin}_{xmin}.las'
        out_tif_path = tif_out_dir + f'/{tif_name}_{ymin}_{xmin}.tif'
        crop_tif_by_bounds(tif_path, out_tif_path, posXmin, posY, posX, posYmin)
        crop_pc_by_bounds(pc_path, out_pc_path, posXmin, posY, posX, posYmin)
        xmin = x + 1
        x += step


if __name__ == "__main__":
    data_dir = f'/mnt/data/davletshina/datasets/Bera_MDE'
    pc_path = f'{data_dir}/KirbyLeafOn2017PointCloudEntireSite.las'
    tif_path = f'{data_dir}/KirbyLeafOn2017RGBNEntireSite.tif'
    splits_dir = 'splits3'

    pc_name = basename(pc_path)[:-4]
    tif_name = basename(tif_path)[:-4]

    pc_out_dir = f'{data_dir}/{splits_dir}/{pc_name}'
    tif_out_dir = f'{data_dir}/{splits_dir}/{tif_name}'
    try:
        os.makedirs(pc_out_dir)
        os.makedirs(tif_out_dir)
    except OSError as e:
        print(e)
        pass

    raster = gdal.Open(tif_path)
    width = raster.RasterXSize
    height = raster.RasterYSize
    print(f'GIVEN IMG: {width}x{height}')

    xoffset, w1, h1, yoffset, w2, h2 = raster.GetGeoTransform()

    # parallel sliding square window
    window = IMG_WIDTH + 1
    step = window

    num_cores = multiprocessing.cpu_count()
    print(f'RUNNING ON {num_cores} CPUs')
    ys = tqdm(range(window - 1, height, step))
    Parallel(n_jobs=num_cores)(delayed(process_row)(y) for y in ys)
