from glob import glob
from os.path import basename, dirname, getsize
import shutil
from tqdm import tqdm


def is_empty(path):
    return getsize(path) < 1500


if __name__ == "__main__":
    splits_dir = "/mnt/data/davletshina/datasets/Bera_MDE/splits"
    empty_dir = "/mnt/data/davletshina/datasets/Bera_MDE/empty"

    prefixes = ["KirbyLeafOn2017PointCloud", "KirbyLeafOff2017PointCloud"]
    # mv empty point clouds and corresponding imgs
    for prefix in prefixes:
        print(prefix)
        for dir in tqdm(glob(f'{splits_dir}/{prefix}*')):
            for pc in glob(f'{dir}/*'):
                if is_empty(pc):
                    name = basename(pc)
                    dir = basename(dirname(pc)).replace("PointCloud", "RGBN")
                    name_no_fmt = name[:-4].replace("PointCloud", "RGBN")
                    if prefix == "KirbyLeafOff2017PointCloud":
                        name_no_fmt = name_no_fmt.replace("e_", "ePCCrop_")
                        dir = f'{dir}PCCrop'
                    shutil.move(f'{splits_dir}/{dir}/{name_no_fmt}.tif', f'{empty_dir}/{name_no_fmt}.tif')
                    shutil.move(pc, f'{empty_dir}/{name}')
