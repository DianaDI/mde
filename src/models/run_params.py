from src.data import IMG_WIDTH, IMG_HEIGHT

RUN_CNT = "85_weighted_huber_grad_norm"
MODEL_DIR = f"run{RUN_CNT}"
FULL_MODEL_SAVING_PATH = f"{MODEL_DIR}/fpn_model_run{RUN_CNT}.pth"
FIG_SAVE_PATH = f"/mnt/data/davletshina/mde/reports/figures/{MODEL_DIR}"

COMMON_PARAMS = {
    'train': True,
    'test': True,
    'load_from_chk': False,
    'chk_point_path': 'run85_l1_loss_w_grad_loss/model_chkp_epoch_18.pth',
    'normalise': True,
    'normalise_type': 'local',  # 'global', 'local'
    'random_seed': 42,
    'num_workers': 10,  # set number of cpu cores for data processing
    'parallel': True,
    'gpu_id': 1,
    'plot_sample': True,
    'test_size': 0.1,
    'save_chk': True,
    'interpolate': True,
    'edge_factor': 0.6,
    'train_gan': False,
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH
}

MODEL_SPECIFIC_PARAMS = {
    'FPNNet': {
        'num_channels': 3,
        'lr': 0.00001,
        'lr_decay': 0.95,
        'batch_size': 46,
        'num_epochs': 20
    },
    'Discriminator': {
        'lr': 0.00001,
        'beta1_d': 0.5,  # Beta1 hyperparam for Adam optimizers
        'nz': 256,  # Size of z latent vector (i.e. size of generator input)
        'ndf': 64,  # Size of feature maps in discriminator
        'loss_weight_gan': 0.2
    }
}
