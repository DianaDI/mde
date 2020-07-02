import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import basename
import shutil
from src.data.transforms import rebin

img_common_dir = "/mnt/data/davletshina/datasets/Bera_MDE/splits2"
dm_dir = "/mnt/data/davletshina/datasets/Bera_MDE/depth_maps2"
movedir = "/mnt/data/davletshina/datasets/Bera_MDE/sparse"

cnt = 0
num_files = len(glob(dm_dir + "/*"))

for file in tqdm(glob(dm_dir + "/*")):
    file_name = basename(file)
    dm = np.load(file, allow_pickle=True)
    dm = rebin(dm, (128, 128))
    zeros = np.sum((dm == 0).astype(int))
    if zeros / dm.size > 0.2:
        cnt += 1
        img_dir = file_name.split('_')[0].replace("DM", "RGBN")
        img_name = file_name.replace("DM", "RGBN").replace("dmp", "tif")
        img_path = f'{img_common_dir}/{img_dir}/{img_name}'
        shutil.move(file, f'{movedir}/{file_name}')
        shutil.move(img_path, f'{movedir}/{img_name}')

print(f'{cnt} out of {num_files} were considered sparse')