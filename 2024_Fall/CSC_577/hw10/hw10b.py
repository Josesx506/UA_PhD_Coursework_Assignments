import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import ConvAutoencoder, SaveBestModel
from tqdm import tqdm

if __name__=="__main__":
    os.system("clear")
    cuda = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    SEED = 577
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    
    # Major Params
    EPOCHS = 30
    batch_size = 128

    
    ####################################### DATA LOADER #######################################
    train_transform = transforms.Compose([transforms.ToTensor(),])

    train_val_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # initialize the NN
    model = ConvAutoencoder()

    criterion = nn.MSELoss() # specify loss function

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    save_best_model = SaveBestModel(path="saved")


    for epoch in range(1, EPOCHS+1):
        # monitor training loss
        train_loss = 0.0
        pbar = tqdm(train_loader)

        # train the model
        for batch_idx, (data, _) in enumerate(pbar):
            images = data.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
        
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        save_best_model(train_loss, epoch, model, optimizer, criterion)
        print(f"Epoch: {epoch:2d} \tTraining Loss: {train_loss:.6f}")
    