import numpy as np
import torch
import os
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiplicativeLR
from src.data.make_dataset import DatadirParser, TrainValTestSplitter, BeraDataset
from src.data import SPLITS_DIR, DEPTH_MAPS
from src.models.mde_net import FPNNet
from src.models.losses import GradLoss, NormalLoss, MaskedL1Loss, HuberLoss, MaskedHuberLoss, L1Loss, CustomHuberLoss
from src.models.util import plot_metrics, save_dict, log_sample
from src.models.run_params import COMMON_PARAMS, MODEL_SPECIFIC_PARAMS, \
    MODEL_DIR, FIG_SAVE_PATH, FULL_MODEL_SAVING_PATH, RUN_CNT
from src.models.eval import eval, calc_loss
from src.models.discriminator import Discriminator, weights_init

# set model type
model_class = FPNNet
gan_class = Discriminator

params = {**COMMON_PARAMS, **MODEL_SPECIFIC_PARAMS[model_class.__name__]}
params_gan = {**MODEL_SPECIFIC_PARAMS[gan_class.__name__]}

device = torch.device("cuda") if params['parallel'] \
    else torch.device("cuda" if torch.cuda.is_available() else "cpu", params['gpu_id'])
loader = transforms.Compose([transforms.ToTensor()])


def save_model_chk(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if params['parallel'] else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def init_data(datadir, labeldir, params):
    random_seed = params['random_seed']
    normalise = params['normalise']
    normalise_type = params['normalise_type']
    interpolate = params['interpolate']
    num_channels = params['num_channels']
    test_size = params['test_size']
    batch_size = params['batch_size']
    num_workers = params['num_workers']

    parser = DatadirParser(datadir, labeldir)
    images, depths = parser.get_parsed()
    splitter = TrainValTestSplitter(images, depths, random_seed=random_seed, test_size=test_size)

    train_ds = BeraDataset(img_filenames=splitter.data_train.image, depth_filenames=splitter.data_train.depth,
                           num_channels=num_channels, normalise=normalise, normalise_type=normalise_type,
                           interpolate=interpolate)
    validation_ds = BeraDataset(img_filenames=splitter.data_val.image, depth_filenames=splitter.data_val.depth,
                                num_channels=num_channels, normalise=normalise, normalise_type=normalise_type,
                                interpolate=interpolate)
    test_ds = BeraDataset(img_filenames=images, depth_filenames=depths,
                          num_channels=num_channels, normalise=normalise, normalise_type=normalise_type,
                          interpolate=interpolate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def init_network(params, params_gan):
    netD = None
    optimizerD = None
    train_gan = params['train_gan']
    ngpu = torch.cuda.device_count()

    modelMDE = FPNNet(num_channels=params['num_channels'])
    # save model architecture
    with open(f'{MODEL_DIR}/network_layers.txt', 'w') as f:
        print(modelMDE, file=f)
    if train_gan:
        netD = Discriminator(ngpu)
        with open(f'{MODEL_DIR}/discrim_layers.txt', 'w') as f:
            print(netD, file=f)

    # wrap into DataParallel to run in several GPUs
    if params['parallel'] and ngpu > 1:
        print(f"Using {ngpu} GPUs")
        modelMDE = nn.DataParallel(modelMDE, list(range(ngpu)))
        if train_gan: netD = nn.DataParallel(netD, list(range(ngpu)))

    modelMDE.to(device)
    optimizer = torch.optim.Adam(modelMDE.parameters(), lr=params['lr'], weight_decay=4e-5)
    lmbda = lambda epoch: params['lr_decay']
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

    total_params = sum(p.numel() for p in modelMDE.parameters())
    print(f'\nNum of parameters in MDE net: {total_params}')

    if train_gan:
        netD.to(device)
        netD.apply(weights_init)
        total_params_d = sum(p.numel() for p in netD.parameters())
        print(f'\nNum of parameters in Discriminator net: {total_params_d}')
        optimizerD = torch.optim.Adam(netD.parameters(), lr=params_gan['lr'], betas=(params_gan['beta1_d'], 0.999))
    return modelMDE, netD, optimizer, optimizerD, scheduler


def train_mde_on_batch(data, model, fig_save_path, epoch, batch_idx, params):
    """
    Train the main MDE model
    :param data: batch data
    :param model: MDE model
    :param fig_save_path:
    :param epoch: current epoch
    :param batch_idx:  current batch id
    :return: loss on batch
    """
    loss, l1_losses, out, target, inp, orig_inp, edges = calc_loss(data, model,
                                                                   l1_criterion, grad_criterion, normal_criterion,
                                                                   device=device,
                                                                   interpolate=params['interpolate'],
                                                                   edge_factor=params['edge_factor'],
                                                                   batch_idx=batch_idx)
    print(f'Epoch {epoch}, batch_idx {batch_idx} train loss: {loss.item()}')
    if params['plot_sample']:
        log_sample(batch_idx, 100, out, target, orig_inp, edges, l1_losses,
                   fig_save_path, epoch, "train")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_on_batch(data, model, fig_save_path, epoch, batch_idx, params):
    loss, l1_losses, out, target, _, orig_inp, edges = calc_loss(data, model,
                                                                 l1_criterion, grad_criterion, normal_criterion,
                                                                 device=device,
                                                                 interpolate=params['interpolate'],
                                                                 edge_factor=params['edge_factor'],
                                                                 batch_idx=batch_idx)
    if params['plot_sample']:
        log_sample(batch_idx, 100, out, target, orig_inp, edges, l1_losses,
                   fig_save_path, epoch, "validate")
    return loss.item()


def train_gan_on_batch(data, model, netD, epoch, batch_idx, params):
    """
    Train MDE network as a generator and separate Discriminator net in GAN fashion
    :param data: batch data
    :param model: MDE net
    :param netD: Discriminator net
    :param epoch: current epoch
    :param batch_idx: current batch id
    :return: losses
    """
    # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Train with all-real batch
    netD.zero_grad()
    dm_inp = Variable(data['depth']).to(device, dtype=torch.float).unsqueeze(1)
    b_size = dm_inp.size(0)

    label = torch.full((b_size,), real_label, device=device)
    output = netD(dm_inp).view(-1)
    errD_real = criterion_bce(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    # Train with all-fake batch
    gen_loss, l1_losses, gen_out, target, inp, orig_inp, edges = calc_loss(data, model, l1_criterion, grad_criterion, normal_criterion,
                                                                           device=device,
                                                                           interpolate=params['interpolate'],
                                                                           edge_factor=params['edge_factor'],
                                                                           batch_idx=batch_idx)

    label.fill_(fake_label)
    output = netD(gen_out.detach()).view(-1)
    errD_fake = criterion_bce(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    # Update G network: maximize log(D(G(z)))
    model.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    output = netD(gen_out).view(-1)
    errG = criterion_bce(output, label)
    gen_loss.backward(retain_graph=True)
    errG = params_gan['loss_weight_gan'] * errG
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizer.step()

    print(f'Epoch {epoch}, batch_idx {batch_idx} train loss: {gen_loss}, gen-discr loss {errG}')
    if batch_idx % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, params['num_epochs'], batch_idx, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    if params['plot_sample']:
        log_sample(batch_idx, 100, gen_out, target, orig_inp, edges, l1_losses,
                   FIG_SAVE_PATH, epoch, "train")
    return gen_loss.item(), errG.item(), errD.item()


def train_on_batch(data, modelMDE, netD, epoch, batch_idx, params):
    loss = 0
    if not train_gan:
        loss = train_mde_on_batch(data, modelMDE, FIG_SAVE_PATH, epoch, batch_idx, params)
    else:
        loss, errG, errD = train_gan_on_batch(data, modelMDE, netD, epoch, batch_idx, params)
        G_losses.append(errG)
        D_losses.append(errD)
    train_loss.append(loss)
    return train_loss, G_losses, D_losses


if __name__ == '__main__':
    # create necessary dirs
    try:
        os.makedirs(FIG_SAVE_PATH)
        os.makedirs(MODEL_DIR)
    except OSError as e:
        print(e)
        pass

    save_dict(params, f'{MODEL_DIR}/{model_class.__name__}_run{RUN_CNT}')

    parallel = params['parallel']
    train_gan = params['train_gan']

    loader_init_fn = lambda worker_id: np.random.seed(params['random_seed'] + worker_id)

    # dataset
    train_loader, val_loader, test_loader = init_data(SPLITS_DIR, DEPTH_MAPS, params)

    # network initialization
    modelMDE, netD, optimizer, optimizerD, scheduler = init_network(params, params_gan)

    # define error criterions
    l1_criterion = MaskedL1Loss()
    grad_criterion = GradLoss()
    normal_criterion = NormalLoss()
    criterion_bce = nn.BCELoss()
    huber = CustomHuberLoss()  # HuberLoss() #MaskedHuberLoss()
    # ssim = SSIM()

    total_train_loss, total_val_loss = [], []
    total_G_loss, total_D_loss = [], []
    first_epoch = 0

    if params['train']:
        print("\nTRAINING STARTING...")
        if params['load_from_chk'] and not train_gan:
            checkpoint = torch.load(params['chk_point_path'])
            if parallel:
                modelMDE.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                modelMDE.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            first_epoch = checkpoint['epoch'] + 1

        # ## Start training
        for epoch in range(first_epoch, params['num_epochs']):
            print(f'====== Epoch {epoch} ======')
            train_loss, valid_loss, G_losses, D_losses = [], [], [], []
            # Train model
            modelMDE.train()
            for batch_idx, data in enumerate(train_loader):
                train_loss, G_losses, D_losses = train_on_batch(data, modelMDE, netD, epoch, batch_idx, params)

            if params['save_chk']:
                save_model_chk(epoch, modelMDE, optimizer, f'{MODEL_DIR}/model_chkp_epoch_{epoch}.pth')
                if train_gan:
                    save_model_chk(epoch, netD, optimizerD, f'{MODEL_DIR}/modelD_chkp_epoch_{epoch}.pth')

            # Validate model
            modelMDE.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    loss = validate_on_batch(data, modelMDE, FIG_SAVE_PATH, epoch, batch_idx, params)
                    print(f'Epoch {epoch}, batch_idx {batch_idx} val loss: {loss}')
                    valid_loss.append(loss)

            # End of epoch actions
            scheduler.step()

            total_train_loss = np.concatenate((total_train_loss, train_loss))
            total_val_loss = np.concatenate((total_val_loss, valid_loss))
            total_G_loss = np.concatenate((total_G_loss, G_losses))
            total_D_loss = np.concatenate((total_D_loss, D_losses))
            plot_metrics(metrics=[train_loss, valid_loss, total_G_loss, total_D_loss],
                         names=["Train losses", "Validation losses", "G_losses", "D_losses"],
                         save_path=FIG_SAVE_PATH, mode=f"train_val_{epoch}")

        plot_metrics(metrics=[total_train_loss, total_val_loss, total_G_loss, total_D_loss],
                     names=["Total Train losses", "Total Validation losses", "Total Train Gen Loss", "Total Train Discr Loss"],
                     save_path=FIG_SAVE_PATH, mode=f"train_val")
        torch.save(modelMDE.state_dict(), FULL_MODEL_SAVING_PATH)

    # Evaluate model
    if params['test']:
        eval(modelMDE,
             test_loader,
             device, False,  # do not interpolate for metrics evaluation
             FULL_MODEL_SAVING_PATH,
             f'{MODEL_DIR}/{model_class.__name__}_run{RUN_CNT}_eval_results',
             FIG_SAVE_PATH,
             plot=True)
