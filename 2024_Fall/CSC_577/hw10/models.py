import os

import torch
import torch.nn as nn
import torch.nn.functional as F


####################################### MODEL #######################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(392, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class OverfitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.group1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(96),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.group2 = nn.Sequential(nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.group3 = nn.Sequential(nn.Conv2d(128, 384, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(384),nn.ReLU())
        self.group4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(384),nn.ReLU())
        self.group5 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, stride=2, padding=2),
                                    nn.BatchNorm2d(128),nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.linear1 = nn.Sequential(nn.Linear(128, 512),nn.ReLU()) #nn.Dropout(0.1),
        self.linear2 = nn.Sequential(nn.Linear(512, 256),nn.ReLU()) #nn.Dropout(0.1),
        self.linear3= nn.Sequential(nn.Linear(256, 10))

    def forward(self, x):
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.conv5 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.conv6 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.conv7 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.conv8 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.conv9 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        ## encode ##
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool3(F.relu(self.conv4(x)))
        
        ## decode ##
        x = self.upsample1(F.relu(self.conv5(x))) 
        x = self.upsample2(F.relu(self.conv6(x)))
        x = self.upsample3(F.relu(self.conv7(x)))
        x = torch.sigmoid(self.conv9(F.relu(self.conv8(x))))
        
        return x

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch"s 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float("inf"),path="outputs"):
        self.best_valid_loss = best_valid_loss
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_path = path
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            # print(f"\nBest validation loss: {self.best_valid_loss}")
            # print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({"epoch": epoch+1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": criterion,}, f"{self.save_path}/best_model.pth")
            
# Resaving the weights of a gpu model to work on CPU
# model = ConvAutoencoder()
# model = model.to(device)
# if device == "cuda":
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = False
#     cudnn.deterministic = True
# model.load_state_dict(torch.load("saved/best_model.pth", weights_only=True), strict=False)
# torch.save(model.module.to("cpu").state_dict(), "saved/best_model_cpu.pth") # ensure .module is used to convert it from nn.DataParallel

