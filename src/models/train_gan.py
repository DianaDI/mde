import numpy as np
import torch
import os
import cv2
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiplicativeLR
import torchvision.utils as vutils
from src.data.make_dataset import DatadirParser, TrainValTestSplitter, BeraDataset
from src.data import SPLITS_DIR, DEPTH_MAPS
from src.models.mde_net import FPNNet
from src.models.losses import GradLoss, NormalLoss, MaskedL1Loss, L1Loss
from src.models.util import plot_metrics, plot_sample, save_dict, imgrad_yx, save_dm
from src.models.run_params import COMMON_PARAMS, MODEL_SPECIFIC_PARAMS, \
    MODEL_DIR, FIG_SAVE_PATH, FULL_MODEL_SAVING_PATH, RUN_CNT
from src.models.ssim import SSIM
from src.models.metrics import root_mean_squared_error
from src.models.discriminator import Discriminator, weights_init

# set model type
model_class = FPNNet

params = {**COMMON_PARAMS, **MODEL_SPECIFIC_PARAMS[model_class.__name__]}

device = torch.device("cuda") if params['parallel'] \
    else torch.device("cuda" if torch.cuda.is_available() else "cpu", params['gpu_id'])
loader = transforms.Compose([transforms.ToTensor()])


def save_model_chk(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def compute_metrics(output, target):
    # mre = mean_relative_error(output, target)
    rmse = root_mean_squared_error(output, target)
    l1, pixel_losses = L1Loss().forward(output, target)
    print(f'RMSE: {rmse}, L1: {l1}')
    return rmse, l1, pixel_losses


def log_sample(cur_batch, plot_every, out, target, inp, edges, pixel_loss, path, epoch, mode):
    if cur_batch % plot_every == 0:
        plot_sample(cv2.merge((inp[0][0, :, :].numpy(),
                               inp[0][1, :, :].numpy(),
                               inp[0][2, :, :].numpy())),
                    out[0][0, :, :].cpu().detach().numpy(),
                    target[0][0, :, :].cpu().detach().numpy(),
                    edges[0][0, :, :].cpu().numpy(),
                    pixel_loss[0][0, :, :].cpu().detach().numpy(),
                    path, epoch, cur_batch, mode)


def get_prediction(data, model):
    orig_inp = data['image'].permute(0, 3, 1, 2)
    inp = Variable(orig_inp).to(device, dtype=torch.float)
    target = Variable(data['depth']).to(device, dtype=torch.float).unsqueeze(1)
    mask = data['mask'].to(device).unsqueeze(1)
    edges = Variable(data['edges']).to(device).unsqueeze(1)
    out = model(inp)
    if not interpolate:
        out = out * mask
        target = target * mask
    return inp, out, target, edges.detach(), orig_inp.detach()


def calc_loss(data, model, l1_criterion, criterion_img, criterion_norm, batch_idx, reg=False):
    inp, out, target, edges, orig_inp = get_prediction(data, model)
    imgrad_true = imgrad_yx(target, device)
    imgrad_out = imgrad_yx(out, device)
    l1_loss, l1_losses = l1_criterion(out, target, edges, device, factor=params['edge_factor'])
    loss_grad = criterion_img(imgrad_out, imgrad_true)
    loss_normal = criterion_norm(imgrad_out, imgrad_true)
    total_loss = l1_loss + loss_grad + 0.5 * loss_normal
    if reg:
        loss_reg = Variable(torch.tensor(0.)).to(device)
        for param in model.parameters():
            loss_reg = loss_reg + param.norm(2)
        total_loss = total_loss + 1e-20 * loss_reg
    if batch_idx % 10 == 0:
        print(f'DM l1-loss: {l1_loss.item()}, '
              f'Loss Grad: {loss_grad.item()},  '
              f'Loss Normal: {loss_normal.item()}')
    return total_loss, l1_losses, out, target, inp, orig_inp, edges


def validate_on_batch(data, model, l1_criterion, criterion_img, criterion_norm, fig_save_path, epoch, batch_idx):
    loss, l1_losses, out, target, inp, orig_inp, edges = calc_loss(data, model, l1_criterion, criterion_img, criterion_norm, batch_idx)
    if params['plot_sample']:
        log_sample(batch_idx, 100, out, target, orig_inp, edges, l1_losses,
                   fig_save_path, epoch, "validate")
    return loss.item()


if __name__ == '__main__':
    try:
        os.makedirs(FIG_SAVE_PATH)
        os.makedirs(MODEL_DIR)
    except OSError as e:
        print(e)
        pass

    save_dict(params, f'{MODEL_DIR}/{model_class.__name__}_run{RUN_CNT}')

    random_seed = params['random_seed']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    normalise = params['normalise']
    normalise_type = params['normalise_type']
    interpolate = params['interpolate']
    num_channels = params['num_channels']

    loader_init_fn = lambda worker_id: np.random.seed(random_seed + worker_id)

    # dataset
    parser = DatadirParser(SPLITS_DIR, DEPTH_MAPS)
    images, depths = parser.get_parsed()
    splitter = TrainValTestSplitter(images, depths, random_seed=random_seed, test_size=params['test_size'])

    train_ds = BeraDataset(img_filenames=splitter.data_train.image, depth_filenames=splitter.data_train.depth,
                           num_channels=num_channels, normalise=normalise, normalise_type=normalise_type,
                           interpolate=interpolate)
    validation_ds = BeraDataset(img_filenames=splitter.data_val.image, depth_filenames=splitter.data_val.depth,
                                num_channels=num_channels, normalise=normalise, normalise_type=normalise_type,
                                interpolate=interpolate)
    test_ds = BeraDataset(img_filenames=splitter.data_test.image, depth_filenames=splitter.data_test.depth,
                          num_channels=num_channels, normalise=normalise, normalise_type=normalise_type,
                          interpolate=interpolate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # network initialization
    model = FPNNet(num_channels=num_channels)
    # Create the Discriminator
    ngpu = torch.cuda.device_count()
    netD = Discriminator(ngpu)

    if params['parallel']:
        if ngpu > 1:
            print(f"Using {ngpu} GPUs")
            model = nn.DataParallel(model, list(range(ngpu)))
            netD = nn.DataParallel(netD, list(range(ngpu)))
    model.to(device)
    netD.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nNum of parameters: {total_params}')

    # apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    l1_criterion = MaskedL1Loss()
    grad_criterion = GradLoss()
    normal_criterion = NormalLoss()

    # Disciminator params
    # Size of z latent vector (i.e. size of generator input)
    nz = 512
    # Size of feature maps in discriminator
    ndf = 64
    # Learning rate for optimizers
    lr_d = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1_d = 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=4e-5)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1_d, 0.999))

    criterion_bce = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(1, 3, 512, 512, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    lmbda = lambda epoch: params['lr_decay']
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    total_train_loss, total_val_loss = [], []
    first_epoch = 0
    if params['train']:
        print("\nTRAINING STARTING...")
        for epoch in range(first_epoch, params['num_epochs']):
            print(f'====== Epoch {epoch} ======')
            train_loss, valid_loss = [], []
            model.train()
            for batch_idx, data in enumerate(train_loader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                dm_inp = Variable(data['depth']).to(device, dtype=torch.float).unsqueeze(1)
                b_size = dm_inp.size(0)
                label = torch.full((b_size,), real_label, device=device)
                # Forward pass real batch through D
                output = netD(dm_inp).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion_bce(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, 3, 512, 512, device=device)
                # Generate fake image batch with G
                fake = model(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion_bce(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                optimizer.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)

                # Calculate G's loss based on this output
                errG = criterion_bce(output, label)
                loss, _, _, _, _, _, _ = calc_loss(data, model, l1_criterion, grad_criterion, normal_criterion, batch_idx)

                loss.backward()
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizer.step()

                print(f'Epoch {epoch}, batch_idx {batch_idx} train loss: {loss}, gen-discr loss {D_G_z2}')
                if batch_idx % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, params['num_epochs'], batch_idx, len(train_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                train_loss.append(loss)
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                # Check how the generator is doing by saving G's output on fixed_noise
                # if (batch_idx % 500 == 0) or ((epoch == params['num_epochs'] - 1) and (batch_idx == len(train_loader) - 1)):
                #     with torch.no_grad():
                #         fake = model(fixed_noise).detach().cpu()
                #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    loss = validate_on_batch(data=data, model=model,
                                             l1_criterion=l1_criterion, criterion_img=grad_criterion, criterion_norm=normal_criterion,
                                             fig_save_path=FIG_SAVE_PATH, epoch=epoch, batch_idx=batch_idx)
                    print(f'Epoch {epoch}, batch_idx {batch_idx} val loss: {loss}')
                    valid_loss.append(loss)

            # enf of epoch actions
            scheduler.step()
            total_train_loss = np.concatenate((total_train_loss, train_loss))
            total_val_loss = np.concatenate((total_val_loss, valid_loss))
            plot_metrics(metrics=[train_loss, valid_loss],
                         names=["Train losses", "Validation losses"],
                         save_path=FIG_SAVE_PATH,
                         mode=f"train_val_{epoch}")
        plot_metrics(metrics=[total_train_loss, total_val_loss],
                     names=["Total Train losses", "Total Validation losses"],
                     save_path=FIG_SAVE_PATH,
                     mode=f"train_val")
        torch.save(model.state_dict(), FULL_MODEL_SAVING_PATH)

    if params['test']:
        print("\nEVALUATION STARTING...")
        model.load_state_dict(torch.load(FULL_MODEL_SAVING_PATH))
        model.eval()
        mre, rmse, l1, l1_range = [], [], [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                inp, out, target, edges, orig_inp = get_prediction(data, model)
                rmse_loss, l1_loss, pixel_losses = compute_metrics(out, target)
                rmse.append(rmse_loss)
                l1.append(l1_loss)
                if params['plot_sample']:
                    log_sample(batch_idx, 50, out, target, orig_inp, edges, pixel_losses, FIG_SAVE_PATH, "", "eval")
                    if batch_idx % 100 == 0:
                        save_dm(out[0][0, :, :].cpu().detach().numpy(), target[0][0, :, :].cpu().detach().numpy(), FIG_SAVE_PATH, batch_idx)
        results = {
            "Mean RMSE Loss": np.mean(rmse),
            "Mean L1 Loss": np.mean(l1)
        }
        for key in results:
            print(f'{key}: {results[key]}')
        save_dict(results, f'{MODEL_DIR}/{model_class.__name__}_run{RUN_CNT}_eval_results')
