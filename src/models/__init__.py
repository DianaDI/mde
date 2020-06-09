# Common
TRAIN = True
TEST = True
RUN_CNT = 5
FULL_MODEL_SAVING_PATH = f"fpn_model_run{RUN_CNT}.pth"
# FULL_MODEL_SAVING_PATH = f"fpn_model.pth"
CHK_MODEL_PATH = "run0/model_chkp_epoch_4.pth"  # specify checkpoint path
LOAD_FROM_CHK = False
MODEL_DIR = f"run{RUN_CNT}"
FIG_SAVE_PATH = f"/mnt/data/davletshina/mde/reports/figures/{MODEL_DIR}"
NORMALIZE = True
RANDOM_SEED = 42
NUM_WORKERS = 10  # set number of cpu cores for images processing
GPU_ID = 0
PLOT_SAMPLE = True

# Model training
LR = 0.00001
LR_DECAY = 0.95
BATCH_SIZE = 12
NUM_EPOCHS = 3
