import numpy as np
import torch
import os
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiplicativeLR
from src.data.make_dataset import DatadirParser, TrainValTestSplitter, BeraDataset
from src.models.mde_net import MDENet
from src.models.util import plot_metrics, plot_sample
from src.models import TRAIN, TEST, \
    CHK_MODEL_PATH, LOAD_FROM_CHK, \
    MODEL_DIR, FIG_SAVE_PATH, FULL_MODEL_SAVING_PATH, \
    NORMALIZE, RANDOM_SEED, LR, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, LR_DECAY, GPU_ID, PLOT_SAMPLE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", GPU_ID)
loader = transforms.Compose([transforms.ToTensor()])


def mean_relative_error(output, target):
    # todo might not be valid with normalization
    target_no_zeros = torch.where(target > 0, target, target + 0.001)
    loss = torch.mean(torch.abs(output - target) / torch.abs(target_no_zeros))
    if loss.item() > 10:
        print(loss.item())
    return loss.item()


def root_mean_squared_error(output, target):
    loss = torch.sqrt(torch.mean((output - target) ** 2))
    return loss.item()


def compute_metrics(output, target):
    mre = mean_relative_error(output, target)
    rmse = root_mean_squared_error(output, target)
    l1 = nn.L1Loss().forward(output, target).item()
    print(f'MRE: {mre}, RMSE: {rmse}, L1: {l1}')
    return mre, rmse, l1


def prepare_var(data):
    inp = Variable(data['image'].permute(0, 3, 1, 2)).to(DEVICE, dtype=torch.float)
    target = Variable(data['depth']).to(DEVICE, dtype=torch.float).unsqueeze(1)
    mask = data['mask'].to(DEVICE).unsqueeze(1)
    return inp, target, mask


def train_on_batch(data, model, criterion, epoch, batch_idx):
    inp, target, mask = prepare_var(data)
    out = model(inp)
    out = out * mask
    target = target * mask
    loss = criterion(out, target)
    if batch_idx % 100 == 0:
        for pred, truth in zip(out, target):
            plot_sample(pred[0, :, :].cpu().detach().numpy(),
                        truth[0, :, :].cpu().detach().numpy(),
                        FIG_SAVE_PATH, epoch, batch_idx, "train")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_on_batch(data, model, criterion, fig_save_path, epoch, batch_idx):
    inp, target, mask = prepare_var(data)
    out = model(inp)
    out = out * mask
    target = target * mask
    loss = criterion(out, target)
    if batch_idx % 10 == 0:
        for pred, truth in zip(out, target):
            plot_sample(pred[0, :, :].cpu(),
                        truth[0, :, :].cpu(),
                        fig_save_path, epoch, batch_idx, "validate")
    return loss.item()


if __name__ == '__main__':
    try:
        os.makedirs(FIG_SAVE_PATH)
        os.makedirs(MODEL_DIR)
    except OSError as e:
        print(e)
        pass

    loader_init_fn = lambda worker_id: np.random.seed(RANDOM_SEED + worker_id)

    # dataset
    parser = DatadirParser()
    images, depths = parser.get_parsed()
    splitter = TrainValTestSplitter(images, depths, random_seed=RANDOM_SEED)

    train_ds = BeraDataset(img_filenames=splitter.data_train.image, depth_filenames=splitter.data_train.depth, normalise=NORMALIZE)
    validation_ds = BeraDataset(img_filenames=splitter.data_val.image, depth_filenames=splitter.data_val.depth, normalise=NORMALIZE)
    test_ds = BeraDataset(img_filenames=splitter.data_test.image, depth_filenames=splitter.data_test.depth, normalise=NORMALIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # network initialization
    model = MDENet().to(DEVICE)
    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    train_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nNum of parameters: {total_params}. Trainable parameters: {train_total_params}')

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=4e-5)
    lmbda = lambda epoch: LR_DECAY
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    total_train_loss = []
    total_val_loss = []
    if TRAIN:
        print("\nTRAINING STARTING...")
        chkp_epoch = 0
        if LOAD_FROM_CHK:
            checkpoint = torch.load(CHK_MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            chkp_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        start_epoch = 0 if not LOAD_FROM_CHK else chkp_epoch
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f'====== Epoch {epoch} ======')
            train_loss, valid_loss = [], []
            loss = 0
            model.train()
            for batch_idx, data in enumerate(train_loader):
                loss = train_on_batch(data, model, criterion, epoch, batch_idx)
                print(f'Epoch {epoch}, batch_idx {batch_idx} train loss: {loss}')
                train_loss.append(loss)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'{MODEL_DIR}/model_chkp_epoch_{epoch}.pth')

            model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    loss = evaluate_on_batch(data, model, criterion, FIG_SAVE_PATH, epoch, batch_idx)
                    print(f'Epoch {epoch}, batch_idx {batch_idx} val loss: {loss}')
                    valid_loss.append(loss)

            # enf of epoch actions
            scheduler.step()
            total_train_loss = np.concatenate((total_train_loss, train_loss))
            total_val_loss = np.concatenate((total_val_loss, valid_loss))
            plot_metrics(metrics=[train_loss, valid_loss], names=["Train losses", "Validation losses"],
                         save_path=FIG_SAVE_PATH, mode=f"train_val_{epoch}")
        plot_metrics(metrics=[total_train_loss, total_val_loss], names=["Total Train losses", "Total Validation losses"],
                     save_path=FIG_SAVE_PATH, mode=f"train_val")
        torch.save(model.state_dict(), FULL_MODEL_SAVING_PATH)

    if TEST:
        print("\nEVALUATION STARTING...")
        model.load_state_dict(torch.load(FULL_MODEL_SAVING_PATH))
        model.eval()
        mre, rmse, l1 = [], [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                inp, target, mask = prepare_var(data)
                out = model(inp)
                out = out * mask
                target = target * mask
                mre_loss, rmse_loss, l1_loss = compute_metrics(out, target)
                mre.append(mre_loss)
                rmse.append(rmse_loss)
                l1.append(l1_loss)
                if PLOT_SAMPLE & (batch_idx % 100) == 0:
                    for pred, truth in zip(out, target):
                        plot_sample(pred[0, :, :].cpu(),
                                    truth[0, :, :].cpu(),
                                    FIG_SAVE_PATH, 0, batch_idx, "eval")
        print(f'Mean MRE Loss: {np.mean(mre)}, Mean RMSE Loss: {np.mean(rmse)}, L1 Loss: {np.mean(l1)}')
