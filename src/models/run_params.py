COMMON_PARAMS = {
    'train': True,
    'test': True,
    'load_from_chk': False,
    'chk_point_path': 'run0/model_chkp_epoch_4.pth',
    'normalise': True,
    'normalise_type': 'local',  # 'global', 'local'
    'random_seed': 123,
    'num_workers': 10,  # set number of cpu cores for images processing
    'gpu_id': 1,
    'plot_sample': True,
    'test_size': 0.1,
    'save_chk': False,
    'interpolate': True
}

MODEL_SPECIFIC_PARAMS = {
    'FPNNet': {
        'lr': 0.00001,
        'lr_decay': 0.95,
        'batch_size': 13,
        'num_epochs': 5
    }
}
