COMMON_PARAMS = {
    'train': True,
    'test': True,
    'load_from_chk': False,
    'chk_point_path': 'run0/model_chkp_epoch_4.pth',
    'normalise': True,
    'random_seed': 42,
    'num_workers': 10,  # set number of cpu cores for images processing
    'gpu_id': 0,
    'plot_sample': True
}

MODEL_SPECIFIC_PARAMS = {
    'FPNNet': {
        'lr': 0.00001,
        'lr_decay': 0.95,
        'batch_size': 12,
        'num_epochs': 4
    }
}
