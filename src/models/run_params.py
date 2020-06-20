COMMON_PARAMS = {
    'train': True,
    'test': True,
    'load_from_chk': False,
    'chk_point_path': 'run0/model_chkp_epoch_4.pth',
    'normalise': True,
    'normalise_type': 'local', # 'global', 'local'
    'random_seed': 42,
    'num_workers': 10,  # set number of cpu cores for images processing
    'gpu_id': 0,
    'plot_sample': True,
    'test_size': 0.1
}

MODEL_SPECIFIC_PARAMS = {
    'FPNNet': {
        'lr': 0.000005,
        'lr_decay': 0.95,
        'batch_size': 12,
        'num_epochs': 3
    }
}
