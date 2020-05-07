from osgeo import gdal
import os
from glob import glob
from os.path import basename
from src.data.splitter import crop_pc_by_bounds, crop_tif_by_bounds


def save_bbs(inp_pc, inp_tif, out_pc, out_tif, tpl, xoffset, yoffset, w1, w2, h1, h2):
    print("=============================================================")
    cnt = 1
    for bb in tpl:
        # convert to world coordinates
        posXmin = xoffset + w1 * bb[0] + h1 * bb[1]
        posYmin = yoffset + w2 * bb[0] + h2 * bb[1]
        posXmax = xoffset + w1 * bb[2] + h1 * bb[3]
        posYmax = yoffset + w2 * bb[2] + h2 * bb[3]

        print("CROPPING SETTINGS:")
        print("min x: ", posXmin)
        print("min y: ", posYmax)
        print("max x: ", posXmax)
        print("max y: ", posYmin)

        crop_tif_by_bounds(inp_tif, out_tif + f'part{cnt}.tif', posXmin, posYmax, posXmax, posYmin)  # y coord is reversed
        crop_pc_by_bounds(inp_pc, out_pc + f'part{cnt}.las', posXmin, posYmax, posXmax, posYmin)
        cnt += 1


def get_4_splits(width, height):
    mid_x = int(width / 2)
    mid_y = int(height / 2)

    return (0, 0, mid_x, mid_y), \
           (mid_x, 0, width, mid_y), \
           (0, mid_y, mid_x, height), \
           (mid_x, mid_y, width, height)


def process_file(tif_file):
    raster = gdal.Open(tif_file)
    width = raster.RasterXSize
    height = raster.RasterYSize
    xoffset, w1, h1, yoffset, w2, h2 = raster.GetGeoTransform()
    splits_coord = get_4_splits(width, height)

    out_pc_path = pc_out_dir + f'/{pc_name}_split{i}_'
    out_tif_path = tif_out_dir + f'/{tif_name}_split{i}_'
    save_bbs(pc_path, tif_file, out_pc_path, out_tif_path, splits_coord, xoffset, yoffset, w1, w2, h1, h2)


if __name__ == "__main__":
    print("STARTING...")
    data_dir = f'/mnt/data/davletshina/datasets/Bera_MDE'
    pc_path = "/mnt/data/davletshina/datasets/Bera_MDE/KirbyLeafOff2017PointCloudEntireSite.las"
    tif_path = "/mnt/data/davletshina/datasets/Bera_MDE/KirbyLeafOff2017RGBNEntireSitePCCrop.tif"

    pc_name = basename(pc_path)[:-4]
    tif_name = basename(tif_path)[:-4]

    pc_out_dir = f'{data_dir}/{pc_name}_binsplit'
    tif_out_dir = f'{data_dir}/{tif_name}_binsplit'
    try:
        os.makedirs(pc_out_dir)
        os.makedirs(tif_out_dir)
    except OSError as e:
        print(e)
        pass

    split_level = 1
    i = 0
    while i < split_level:
        if i == 0:
            # process initial main tif
            process_file(tif_path)
        else:
            # look in saved folder for all tif files
            for tif in glob(f'{tif_out_dir}/*.tif'):
                process_file(tif)
            # remove prev level splits
            if i > 1:
                for old_split in glob(f'{data_dir}/{tif_out_dir}/{tif_name}_split{i - 1}*.tif'):
                    os.remove(old_split)
        i += 1
