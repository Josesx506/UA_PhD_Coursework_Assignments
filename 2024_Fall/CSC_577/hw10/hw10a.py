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

    ####################################### MODEL #######################################
    # class BasicBlock(nn.Module):
    #     expansion = 1

    #     def __init__(self, in_planes, planes, stride=1):
    #         super(BasicBlock, self).__init__()
    #         self.conv1 = nn.Conv2d(
    #             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
    #         )
    #         self.bn1 = nn.BatchNorm2d(planes)
    #         self.conv2 = nn.Conv2d(
    #             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
    #         )
    #         self.bn2 = nn.BatchNorm2d(planes)

    #         self.shortcut = nn.Sequential()
    #         if stride != 1 or in_planes != self.expansion * planes:
    #             self.shortcut = nn.Sequential(
    #                 nn.Conv2d(
    #                     in_planes,
    #                     self.expansion * planes,
    #                     kernel_size=1,
    #                     stride=stride,
    #                     bias=False,
    #                 ),
    #                 nn.BatchNorm2d(self.expansion * planes),
    #             )

    #     def forward(self, x):
    #         out = F.relu(self.bn1(self.conv1(x)))
    #         out = self.bn2(self.conv2(out))
    #         out += self.shortcut(x)
    #         out = F.relu(out)
    #         return out


    # class ResNet(nn.Module):
    #     def __init__(self, block, num_blocks, num_classes=10):
    #         super(ResNet, self).__init__()
    #         self.in_planes = 64

    #         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #         self.bn1 = nn.BatchNorm2d(64)
    #         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    #         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    #         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    #         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    #         self.linear = nn.Linear(512 * block.expansion, num_classes)

    #     def _make_layer(self, block, planes, num_blocks, stride):
    #         strides = [stride] + [1] * (num_blocks - 1)
    #         layers = []
    #         for stride in strides:
    #             layers.append(block(self.in_planes, planes, stride))
    #             self.in_planes = planes * block.expansion
    #         return nn.Sequential(*layers)

    #     def forward(self, x):
    #         out = F.relu(self.bn1(self.conv1(x)))
    #         out = self.layer1(out)
    #         out = self.layer2(out)
    #         out = self.layer3(out)
    #         out = self.layer4(out)
    #         out = F.avg_pool2d(out, 4)
    #         out = out.view(out.size(0), -1)
    #         out = self.linear(out)
    #         return out


    # def ResNet18():
    #     return ResNet(BasicBlock, [2, 2, 2, 2])
    
    # class SimpleCNN(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv1 = nn.Conv2d(3, 2, 5)
    #         self.pool = nn.MaxPool2d(2, 2)
    #         self.fc1 = nn.Linear(392, 10)

    #     def forward(self, x):
    #         x = self.pool(F.relu(self.conv1(x)))
    #         x = torch.flatten(x, 1)
    #         x = self.fc1(x)
    #         return x
    
    # class OverfitCNN(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.group1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
    #                                     nn.BatchNorm2d(96),nn.ReLU(),
    #                                     nn.MaxPool2d(kernel_size = 2, stride = 2))
    #         self.group2 = nn.Sequential(nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
    #                                     nn.BatchNorm2d(128),nn.ReLU(),
    #                                     nn.MaxPool2d(kernel_size = 2, stride = 2))
    #         self.group3 = nn.Sequential(nn.Conv2d(128, 384, kernel_size=3, stride=2, padding=1),
    #                                     nn.BatchNorm2d(384),nn.ReLU())
    #         self.group4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),
    #                                     nn.BatchNorm2d(384),nn.ReLU())
    #         self.group5 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, stride=2, padding=2),
    #                                     nn.BatchNorm2d(128),nn.ReLU(),
    #                                     nn.MaxPool2d(kernel_size = 2, stride = 2))
    #         self.linear1 = nn.Sequential(nn.Linear(128, 512),nn.ReLU()) #nn.Dropout(0.1),
    #         self.linear2 = nn.Sequential(nn.Linear(512, 256),nn.ReLU()) #nn.Dropout(0.1),
    #         self.linear3= nn.Sequential(nn.Linear(256, 10))

    #     def forward(self, x):
    #         x = self.group1(x)
    #         x = self.group2(x)
    #         x = self.group3(x)
    #         x = self.group4(x)
    #         x = self.group5(x)
    #         x = x.reshape(x.size(0), -1)
    #         x = self.linear1(x)
    #         x = self.linear2(x)
    #         x = self.linear3(x)
    #         return x

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

    