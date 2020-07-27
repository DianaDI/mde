RUN_CNT = 72
MODEL_DIR = f"run{RUN_CNT}"
FULL_MODEL_SAVING_PATH = f"{MODEL_DIR}/fpn_model_run{RUN_CNT}.pth"
FIG_SAVE_PATH = f"/mnt/data/davletshina/mde/reports/figures/{MODEL_DIR}"

COMMON_PARAMS = {
    'train': True,
    'test': True,
    'load_from_chk': False,
    'chk_point_path': 'run0/model_chkp_epoch_4.pth',
    'normalise': True,
    'normalise_type': 'local',  # 'global', 'local'
    'random_seed': 123,
    'num_workers': 3,  # set number of cpu cores for images processing
    'parallel': True,
    'gpu_id': 1,
    'plot_sample': True,
    'test_size': 0.1,
    'save_chk': False,
    'interpolate': True,
    'edge_factor': 0.6
}

MODEL_SPECIFIC_PARAMS = {
    'FPNNet': {
        'num_channels': 3,
        'lr': 0.0001,
        'lr_decay': 0.95,
        'batch_size': 10,
        'num_epochs': 6
    }
}
