import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from src.data.make_dataset import DatadirParser, TrainValTestSplitter, BeraDataset
from src.models.mde_net import MDENet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
loader = transforms.Compose([transforms.ToTensor()])

if __name__ == '__main__':

    random_seed = 42
    # set number of cpu cores for images processing
    num_workers = 4
    loader_init_fn = lambda worker_id: np.random.seed(random_seed + worker_id)

    lr = 0.00001
    batch_size = 4
    num_epochs = 10
    # dataset
    parser = DatadirParser()
    images, depths = parser.get_parsed()
    splitter = TrainValTestSplitter(images, depths)

    train = BeraDataset(img_filenames=splitter.data_train.image, depth_filenames=splitter.data_train.depth)
    validation = BeraDataset(img_filenames=splitter.data_val.image, depth_filenames=splitter.data_val.depth)
    test = BeraDataset(img_filenames=splitter.data_test.image, depth_filenames=splitter.data_test.depth)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # network initialization
    model = MDENet().to(DEVICE)
    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    train_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nNum of parameters: {total_params}. Trainable parameters: {train_total_params}')

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)

    print("\nTRAINING STARTING...")
    for i in range(num_epochs):
        train_loss, valid_loss = [], []
        # training part
        model.train()
        print(f'====== Epoch {i} ======')
        for batch_idx, data in enumerate(train_loader):
            inp = Variable(data['image'].permute(0, 3, 1, 2)).to(DEVICE, dtype=torch.float)
            target = Variable(data['depth']).to(DEVICE, dtype=torch.float).unsqueeze(1)
            mask = data['mask'].to(DEVICE)
            out = model(inp)
            loss = criterion(out * mask, target * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {i}, batch_idx {batch_idx} train loss: {loss.item()}')
            train_loss.append(loss.item())

        # evaluation part
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inp = Variable(data['image'].permute(0, 3, 1, 2)).to(DEVICE, dtype=torch.float)
                target = Variable(data['depth']).to(DEVICE, dtype=torch.float).unsqueeze(1)
                mask = data['mask'].to(DEVICE)
                out = model(inp)
                loss = criterion(out * mask, target * mask)
                print(f'Epoch {i} validation l1-loss: {loss.item()}')
                valid_loss.append(loss.item())

        print(f'Epoch: {i} Training Loss: {np.mean(train_loss)}, Valid Loss: {np.mean(valid_loss)}')

    SAVING_PATH = "."
    torch.save(model.state_dict(), SAVING_PATH)
