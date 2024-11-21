import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pickle_blosc import pickle
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from torchsummary import summary
from models import SimpleCNN,OverfitCNN,ResNet18

from tqdm import tqdm

classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck",)

if __name__=="__main__":
    os.system("clear")
    cuda = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    SEED = 577
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # Major Params
    k_folds = 10
    EPOCHS = 12
    batch_size = 128

    ####################################### DATA LOADER #######################################
    train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomCrop(size=(32, 32)),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

    train_val_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    # trainloader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    ####################################### UTILS #######################################
    
    def reset_weights(m):
        """
            Try resetting model weights to avoid
            weight leakage.
        """
        for layer in m.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def get_lr(optimizer):
        """"
        for tracking how your learning rate is changing throughout training
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    ####################################### TRAINING #######################################
    # Print the model architecture
    # summary(net, input_size=(3, 32, 32))

    architectures = {
        "OverfitModel": OverfitCNN(), # Overfit with no dropout
        "SimpleCNN": SimpleCNN(),
        "BestResNet18": ResNet18(),
    }

    all_results = {
        "SimpleCNN": {},
        "OverfitModel": {},
        "BestResNet18": {}}


    for arch_name, model in architectures.items():
        print(f"\nEvaluating architecture: {arch_name}")
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)

        folds_mean_train_acc = []
        folds_mean_train_loss = []
        folds_mean_val_acc = []
        folds_mean_val_loss = []
        folds_mean_test_acc = []
        folds_mean_test_loss = []
        

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_dataset)):
            print(f"Fold {fold + 1}/{k_folds}")
            train_subsampler = torch.utils.data.Subset(train_val_dataset,train_idx)
            val_subsampler = torch.utils.data.Subset(train_val_dataset,val_idx)
            # print(len(train_idx),len(val_idx))

            trainloader = torch.utils.data.DataLoader(train_subsampler, batch_size=batch_size)
            valloader = torch.utils.data.DataLoader(val_subsampler, batch_size=batch_size)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

            net = model
            net.apply(reset_weights)  # Prevent weight leakage across folds
            net = net.to(device)
            if device == "cuda":
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = False
                cudnn.deterministic = True

            optimizer = optim.SGD(net.parameters(), lr=0.03, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler("cuda")

            # Learning Rate Finder
            lr_finder = LRFinder(net, optimizer, criterion, device=device)
            lr_finder.range_test(trainloader, end_lr=10, num_iter=200, step_mode="exp")
            # lr_finder.plot()   # to inspect the loss-learning rate graph
            lr_finder.reset()  # to reset the model and optimizer to their initial state

            if arch_name == "BestResNet18":
                max_lr = 0.1
            # elif arch_name == "AdvancedModel":
            #     max_lr = 0.001
            else:
                max_lr = 0.01

            # Learning Rate Scheduler - OneCycleLR
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=len(trainloader),
                epochs=EPOCHS,
                pct_start=0.25,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy="linear",
            )


            def train(model, device, train_loader, optimizer, epoch, scheduler, criterion, output_loss, output_lr, output_acc):
                model.train()
                pbar = tqdm(train_loader)
                correct = 0
                processed = 0
                for batch_idx, (data, target) in enumerate(pbar):
                    # get samples
                    data, target = data.to(device), target.to(device)

                    # Init
                    optimizer.zero_grad()
                    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation 
                    # because PyTorch accumulates the gradients on subsequent backward passes.
                    # Because of this, when you start your training loop, ideally you should zero out the 
                    # gradients so that you do the parameter update correctly.

                    # Predict
                    with torch.amp.autocast("cuda"):
                        y_pred = model(data)
                        
                        # Calculate loss
                        loss = criterion(y_pred, target)
                        output_loss.append(loss.item())
                        output_lr.append(get_lr(optimizer))

                    # Backpropagation
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # Update pbar-tqdm
                    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    processed += len(data)

                    pbar.set_description(desc=f"Loss={loss.item():.4f} LR={get_lr(optimizer):.4f} "+\
                                        f"Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}")
                    output_acc.append(100 * correct / processed)


            def test(model, device, test_loader, criterion, output_loss, output_acc, name="Test"):
                model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        test_loss += criterion(output, target).item()   # sum up batch loss
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(test_loader.dataset)
                output_loss.append(test_loss)

                print(f"\n{name} set: Average loss: {test_loss:.4f}, "+\
                    f"Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)\n")

                output_acc.append(100.0 * correct / len(test_loader.dataset))
            
            # Track metrics
            train_loss_per_epoch = []
            train_lr_per_epoch = []
            train_acc_per_epoch = []
            val_loss_per_epoch = []
            val_acc_per_epoch = []
            test_loss_per_epoch = []
            test_acc_per_epoch = []
            
            for epoch in range(EPOCHS):
                print("EPOCH:", epoch)
                train(net, device, trainloader, optimizer, epoch, scheduler, criterion,
                      train_loss_per_epoch, train_lr_per_epoch, train_acc_per_epoch)
                test(net, device, valloader, criterion, val_loss_per_epoch, val_acc_per_epoch, name="Validation") # validation
                test(net, device, testloader, criterion, test_loss_per_epoch, test_acc_per_epoch)
            
            fold_epoch = {
                f"train_loss": train_loss_per_epoch,
                f"train_lr": train_lr_per_epoch,
                f"train_acc": train_acc_per_epoch,
                f"val_loss": val_loss_per_epoch,
                f"val_acc": val_acc_per_epoch,
                f"test_loss": test_loss_per_epoch,
                f"test_acc": test_acc_per_epoch,
            }
            all_results[arch_name][f"k{fold+1}_epochs"] = fold_epoch
            
            folds_mean_train_acc.append(np.mean(train_acc_per_epoch))
            folds_mean_train_loss.append(np.mean(train_loss_per_epoch))
            folds_mean_val_acc.append(np.mean(val_acc_per_epoch))
            folds_mean_val_loss.append(np.mean(val_loss_per_epoch))
            folds_mean_test_acc.append(np.mean(test_acc_per_epoch))
            folds_mean_test_loss.append(np.mean(test_loss_per_epoch))

            save_models = "saved"
            if not os.path.exists(save_models):
                os.makedirs(save_models)
            
            PATH = f"./{save_models}/{arch_name}_k{fold+1}_cifar_net.pth"
            torch.save(net.state_dict(), PATH)

            # print(json.dumps(all_results[arch_name], default=str, indent=4))

        # Calculate averages and standard errors
        fold_summary = {
            "mean_fold_val_acc": folds_mean_val_acc,
            "mean_fold_val_loss": folds_mean_val_loss,
            "mean_fold_test_acc": folds_mean_test_acc,
            "mean_fold_test_loss": folds_mean_test_loss,
            "mean_train_acc": np.mean(folds_mean_train_acc),
            "train_std_error": np.std(folds_mean_train_acc) / np.sqrt(k_folds),
            "mean_val_acc": np.mean(folds_mean_val_acc),
            "val_std_error": np.std(folds_mean_val_acc) / np.sqrt(k_folds),
            "mean_test_acc": np.mean(folds_mean_test_acc),
            "test_std_error": np.std(folds_mean_test_acc) / np.sqrt(k_folds)
        }

        all_results[arch_name]["summary"] = fold_summary
        pickle(all_results, f"{arch_name}_metrics.pkl")

        # print(json.dumps(all_results[arch_name], default=str, indent=4))

    
    pickle(all_results, "metrics.pkl")

    