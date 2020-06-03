import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiplicativeLR
from src.data.make_dataset import DatadirParser, TrainValTestSplitter, BeraDataset
from src.models.mde_net import MDENet
from src.models.util import plot_metrics, plot_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
loader = transforms.Compose([transforms.ToTensor()])


def mean_relative_error(output, target):
    loss = torch.mean((output - target) / target)
    return loss


def root_mean_squared_error(output, target):
    loss = torch.sqrt(torch.mean((output - target) ** 2))
    return loss


def compute_metrics(output, target):
    mre = mean_relative_error(output, target)
    rmse = root_mean_squared_error(output, target)
    l1 = nn.L1Loss(output, target)
    print(f'MRE: {mre}, RMSE: {rmse}, L1: {l1}')
    return mre, rmse, l1


def prepare_var(data):
    inp = Variable(data['image'].permute(0, 3, 1, 2)).to(DEVICE, dtype=torch.float)
    target = Variable(data['depth']).to(DEVICE, dtype=torch.float).unsqueeze(1)
    mask = data['mask'].to(DEVICE).unsqueeze(1)
    return inp, target, mask


def train_on_batch(data, model, criterion, batch_idx):
    inp, target, mask = prepare_var(data)
    out = model(inp)
    out = out * mask
    target = target * mask
    loss = criterion(out, target)
    if batch_idx % 100 == 0:
        for pred, truth in zip(out, target):
            plot_sample(pred[0, :, :].cpu().detach().numpy(),
                        truth[0, :, :].cpu().detach().numpy(),
                        fig_save_path, batch_idx, "train")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_on_batch(data, model, criterion, fig_save_path, batch_idx):
    inp, target, mask = prepare_var(data)
    out = model(inp)
    out = out * mask
    target = target * mask
    loss = criterion(out, target)
    if batch_idx % 10 == 0:
        for pred, truth in zip(out, target):
            plot_sample(pred[0, :, :].cpu(),
                        truth[0, :, :].cpu(),
                        fig_save_path, batch_idx, "validate")
    return loss.item()


if __name__ == '__main__':

    TRAIN = True
    TEST = True
    SAVING_PATH = "fpn_model.pth"
    fig_save_path = "/mnt/data/davletshina/mde/reports/figures"

    random_seed = 42
    # set number of cpu cores for images processing
    num_workers = 8
    loader_init_fn = lambda worker_id: np.random.seed(random_seed + worker_id)

    lr = 0.0001
    batch_size = 12
    num_epochs = 3

    # dataset
    parser = DatadirParser()
    images, depths = parser.get_parsed()
    splitter = TrainValTestSplitter(images, depths, random_seed=random_seed)

    train_ds = BeraDataset(img_filenames=splitter.data_train.image, depth_filenames=splitter.data_train.depth)
    validation_ds = BeraDataset(img_filenames=splitter.data_val.image, depth_filenames=splitter.data_val.depth)
    test_ds = BeraDataset(img_filenames=splitter.data_test.image, depth_filenames=splitter.data_test.depth)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # network initialization
    model = MDENet().to(DEVICE)
    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    train_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nNum of parameters: {total_params}. Trainable parameters: {train_total_params}')

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)
    lmbda = lambda epoch: 0.95
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    if TRAIN:
        print("\nTRAINING STARTING...")
        for epoch in range(num_epochs):
            print(f'====== Epoch {epoch} ======')
            train_loss, valid_loss = [], []
            model.train()
            for batch_idx, data in enumerate(train_loader):
                loss = train_on_batch(data, model, criterion, batch_idx)
                print(f'Epoch {epoch}, batch_idx {batch_idx} train loss: {loss}')
                train_loss.append(loss)

            model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    loss = evaluate_on_batch(data, model, criterion, fig_save_path, batch_idx)
                    print(f'Epoch {epoch}, batch_idx {batch_idx} val loss: {loss}')
                    valid_loss.append(loss)
            scheduler.step()
            plot_metrics(metrics=[train_loss, valid_loss], names=["Train losses", "Validation losses"],
                         save_path=fig_save_path, mode=f"train_val_{epoch}")
        torch.save(model.state_dict(), SAVING_PATH)

    if TEST:
        model = torch.load(SAVING_PATH)
        model.eval()
        mre, rmse, l1 = [], [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                inp, target, mask = prepare_var(data)
                out = model(inp)
                mre_loss, rmse_loss, l1_loss = compute_metrics(out * mask, target * mask)
                mre.append(mre)
                rmse.append(rmse)
                l1.append(l1_loss)
                if batch_idx % 100 == 0:
                    for pred, truth in zip(out, target):
                        plot_sample(pred[0, :, :].cpu(),
                                    truth[0, :, :].cpu(),
                                    fig_save_path, batch_idx, "eval")
        plot_metrics([mre, rmse, l1], ["MRE", "RMSE", "L1"], fig_save_path, mode="eval")
        print(f'Mean MRE Loss: {np.mean(mre)}, RMSE Loss: {np.mean(rmse)}, L1 Loss: {np.mean(l1)}')
