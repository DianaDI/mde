from glob import glob
from os.path import basename
from tqdm import tqdm
import json
import pandas as pd

res_dir = "/mnt/data/davletshina/mde/src/models"
res_df = pd.DataFrame()

not_needed = ['train', 'test', 'chk_point_path', 'num_workers', 'gpu_id', 'save_chk']

for dir in tqdm(glob(f'{res_dir}/run*')):
    run_num = basename(dir)
    if "run_params" in run_num:
        continue
    res = {'run': run_num}
    for j in glob(f'{dir}/*.json'):
        with open(j) as json_file:
            data = json.load(json_file)
            res.update(data)
    for key in not_needed:
        try:
            del res[key]
        except KeyError:
            pass
    cur = pd.DataFrame(res, index=[0])
    res_df = pd.concat([res_df, cur])

path_to_save = f'{res_dir}/all_res.csv'
print(f'Saving to {path_to_save}')
res_df.to_csv(path_to_save, index=False)
