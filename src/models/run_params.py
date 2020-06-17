COMMON_PARAMS = {
    'train': True,
    'test': True,
    'load_from_chk': False,
    'chk_point_path': 'run0/model_chkp_epoch_4.pth',
    'normalise': True,
    'normalise_type': 'local', # 'global', 'local'
    'random_seed': 142,
    'num_workers': 10,  # set number of cpu cores for images processing
    'gpu_id': 1,
    'plot_sample': True
}

MODEL_SPECIFIC_PARAMS = {
    'FPNNet': {
        'lr': 0.0001,
        'lr_decay': 0.85,
        'batch_size': 2,
        'num_epochs': 3
    }
}
