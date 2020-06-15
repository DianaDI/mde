import numpy as np
import torch
import os
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiplicativeLR
from src.data.make_dataset import DatadirParser, TrainValTestSplitter, BeraDataset
from src.models.mde_net import FPNNet
from src.models.util import plot_metrics, plot_sample, save_run_params
from src.models import MODEL_DIR, FIG_SAVE_PATH, FULL_MODEL_SAVING_PATH, RUN_CNT
from src.models.run_params import COMMON_PARAMS, MODEL_SPECIFIC_PARAMS

# set model type
model_class = FPNNet

params = {**COMMON_PARAMS, **MODEL_SPECIFIC_PARAMS[model_class.__name__]}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", params['gpu_id'])
loader = transforms.Compose([transforms.ToTensor()])


def mean_relative_error(output, target):
    # todo might not be valid with normalization
    target_no_zeros = torch.where(target > 0, target, target + 0.001)
    loss = torch.mean(torch.abs(output - target) / torch.abs(target_no_zeros))
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
    range = Variable(data['range']).to(DEVICE)
    return inp, target, mask, range


def log_sample(cur_batch, plot_every, out, target, path, epoch, mode):
    if cur_batch % plot_every == 0:
        plot_sample(out[0][0, :, :].cpu().detach().numpy(),
                    target[0][0, :, :].cpu().detach().numpy(),
                    path, epoch, cur_batch, mode)


def train_on_batch(data, model, criterion, fig_save_path, epoch, batch_idx):
    inp, target, mask, range = prepare_var(data)
    out, out_range = model(inp), 0
    out_masked = out * mask
    target_masked = target * mask
    loss = criterion(out_masked, target_masked)
    # loss_range = criterion(out_range, range)
    # print(f'DM loss: {loss.item()}, Range loss: {loss_range.item()}')
    # loss_reg = Variable(torch.tensor(0.)).to(DEVICE)
    # for param in model.parameters():
    #     loss_reg = loss_reg + param.norm(2)
    # loss = loss + loss_range + 0.0001 * loss_reg
    if params['plot_sample']:
        log_sample(batch_idx, 10, out_masked, target_masked, fig_save_path, epoch, "train")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_on_batch(data, model, criterion, fig_save_path, epoch, batch_idx):
    inp, target, mask, range = prepare_var(data)
    out, out_range = model(inp), 0
    out_masked = out * mask
    target_masked = target * mask
    loss = criterion(out_masked, target_masked)
    # loss_range = criterion(out_range, range)
    # loss = loss + loss_range
    if params['plot_sample']:
        log_sample(batch_idx, 10, out_masked, target_masked, fig_save_path, epoch, "validate")
    return loss.item()


if __name__ == '__main__':
    try:
        os.makedirs(FIG_SAVE_PATH)
        os.makedirs(MODEL_DIR)
    except OSError as e:
        print(e)
        pass

    save_run_params(params, f'{MODEL_DIR}/{model_class.__name__}_run{RUN_CNT}')

    random_seed = params['random_seed']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    normalise = params['normalise']
    normalise_type = params['normalise_type']

    loader_init_fn = lambda worker_id: np.random.seed(random_seed + worker_id)

    # dataset
    parser = DatadirParser()
    images, depths = parser.get_parsed()
    splitter = TrainValTestSplitter(images, depths, random_seed=random_seed)

    train_ds = BeraDataset(img_filenames=splitter.data_train.image, depth_filenames=splitter.data_train.depth,
                           normalise=normalise, normalise_type=normalise_type)
    validation_ds = BeraDataset(img_filenames=splitter.data_val.image, depth_filenames=splitter.data_val.depth,
                                normalise=normalise, normalise_type=normalise_type)
    test_ds = BeraDataset(img_filenames=splitter.data_test.image, depth_filenames=splitter.data_test.depth,
                          normalise=normalise, normalise_type=normalise_type)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # network initialization
    model = FPNNet().to(DEVICE)
    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    train_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nNum of parameters: {total_params}. Trainable parameters: {train_total_params}')

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=4e-5)
    lmbda = lambda epoch: params['lr_decay']
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    total_train_loss, total_val_loss = [], []
    first_epoch = 0
    if params['train']:
        print("\nTRAINING STARTING...")
        if params['load_from_chk']:
            checkpoint = torch.load(params['chk_point_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            first_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        for epoch in range(first_epoch, params['num_epochs']):
            print(f'====== Epoch {epoch} ======')
            train_loss, valid_loss = [], []
            loss = 0
            model.train()
            for batch_idx, data in enumerate(train_loader):
                loss = train_on_batch(data, model, criterion, FIG_SAVE_PATH, epoch, batch_idx)
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

    if params['test']:
        print("\nEVALUATION STARTING...")
        model.load_state_dict(torch.load(FULL_MODEL_SAVING_PATH))
        model.eval()
        mre, rmse, l1 = [], [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                inp, target, mask, range = prepare_var(data)
                out, out_range = model(inp), 0
                out_masked = out * mask
                target_masked = target * mask
                mre_loss, rmse_loss, l1_loss = compute_metrics(out_masked, target_masked)
                mre.append(mre_loss)
                rmse.append(rmse_loss)
                l1.append(l1_loss)
                if params['plot_sample']:
                    log_sample(batch_idx, 100, out_masked, target_masked, FIG_SAVE_PATH, 0, "eval")
        print(f'Mean MRE Loss: {np.mean(mre)}, Mean RMSE Loss: {np.mean(rmse)}, L1 Loss: {np.mean(l1)}')
