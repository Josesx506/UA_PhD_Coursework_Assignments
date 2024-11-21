import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import ConvAutoencoder, ResNet18, SimpleCNN
from pickle_blosc import unpickle
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

np.random.seed(576)

sv_fl = "output"
if not os.path.exists(sv_fl):
    os.makedirs(sv_fl)

batch_size= 128
test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

fol = "metrics"
underfit = unpickle(f"{fol}/SimpleCNN_metrics.pkl")
underfit = underfit["SimpleCNN"]
bestfit = unpickle(f"{fol}/BestResNet18_metrics.pkl")
bestfit = bestfit["BestResNet18"]
overfit = unpickle(f"{fol}/OverfitModel_metrics.pkl")
overfit = overfit["AdvancedModel"]

# Epoch summary
undft_k1 = underfit["k1_epochs"]
bstft_k1 = bestfit["k1_epochs"]
ovrft_k1 = overfit["k1_epochs"]

# Fold summary
undft_smy = underfit["summary"]
bstft_smy = bestfit["summary"]
ovrft_smy = overfit["summary"]

# Fold size, batch size, epochs
fs,bs,epch = 45000,128,12
total_epochs = len(undft_k1["train_loss"]) / (fs/bs)
trx = np.linspace(0, total_epochs, len(undft_k1["train_loss"]))
total_epochs = len(undft_k1["val_loss"])
vlx = np.linspace(0, total_epochs, len(undft_k1["val_loss"]))
total_epochs = len(undft_k1["test_loss"])
tsx = np.linspace(0, total_epochs, len(undft_k1["test_loss"]))


ovr_mtr = {}
kynms = ["mean_train_acc", "train_std_error", "mean_val_acc", "val_std_error", "mean_test_acc", "test_std_error"]
# ovr_mtr["Rows"] = 
ovr_mtr["Underfit"] = [undft_smy[ky] for ky in kynms]
ovr_mtr["Best fit"] = [bstft_smy[ky] for ky in kynms]
ovr_mtr["Overfit"] = [ovrft_smy[ky] for ky in kynms]
ovr_mtr = pd.DataFrame(ovr_mtr,index=["train_acc", "train_std_error", "val_acc", "val_std_error", "test_acc", "test_std_error"])
ovr_mtr = ovr_mtr.round(2)
print(ovr_mtr)




# fold 1 metrics
with plt.rc_context({"font.family": "Times New Roman","font.size":11}):
    # Val and test loss
    fig,axs = plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=False)
    lw,fs = 2,14
    axs[0].plot(vlx,undft_k1["val_loss"],lw=lw,label="val")
    axs[0].plot(tsx,undft_k1["test_loss"],lw=lw,label="test")
    axs[1].plot(vlx,bstft_k1["val_loss"],lw=lw,label="val")
    axs[1].plot(tsx,bstft_k1["test_loss"],lw=lw,label="test")
    axs[2].plot(vlx,ovrft_k1["val_loss"],lw=lw,label="val")
    axs[2].plot(tsx,ovrft_k1["test_loss"],lw=lw,label="test")
    axs[0].set_title("Underfit",size=fs-2)
    axs[1].set_title("Best fit",size=fs-2)
    axs[2].set_title("Overfit",size=fs-2)
    for i,ax in enumerate(axs): 
        if i in [1,2]:
            ax.set_ylim(0.005,0.0125)
        ax.legend(), ax.set_xlabel("epochs",size=fs-1), ax.margins(x=0.01)
    fig.suptitle("Loss Trends for fold(k=1)",size=fs)
    plt.savefig(f"{sv_fl}/val_test_loss_k1.png",bbox_inches="tight",dpi=200)
    plt.close()

    # training loss
    fig,axs = plt.subplots(1,1,figsize=(5,4))
    lw,fs = 2,14
    axs.plot(trx,undft_k1["train_loss"],lw=lw/2,label="underfit")
    axs.plot(trx,bstft_k1["train_loss"],lw=lw/2,label="best fit",alpha=0.85)
    axs.plot(trx,ovrft_k1["train_loss"],lw=lw/2,label="overfit",alpha=0.85)
    axs.legend(fontsize=fs-2), axs.set_xlabel("epochs",size=fs-1), axs.margins(x=0.01)
    axs.set_title("Training Loss Trends for fold(k=1)",size=fs)
    plt.savefig(f"{sv_fl}/training_loss_k1.png",bbox_inches="tight",dpi=200)
    plt.close()

    # training accuracy
    fig,axs = plt.subplots(1,1,figsize=(5,4))
    lw,fs = 2,14
    axs.plot(trx,ovrft_k1["train_acc"],lw=lw/2,label="train")
    axs.plot(tsx,ovrft_k1["test_acc"],lw=lw/2,label="test")
    axs.axvline(3, ls="--", c="k")
    axs.legend(fontsize=fs-2,loc="lower right"), axs.margins(x=0.01)
    axs.set_xlabel("epochs",size=fs-1), axs.set_ylabel("Acc. (%)",size=fs-1)
    axs.set_title("Training accuracy for overfit model [fold (k=1)]",size=fs)
    plt.savefig(f"{sv_fl}/training_acc_k1.png",bbox_inches="tight",dpi=200)
    plt.close()


# print(undft_smy.keys())


def inverse_normalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std).add_(mean)
    return tensor

def imshow(img):
    inverse_normalize(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_model(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predi = torch.max(output, 1)
            pred = output.argmax(dim=1, keepdim=True)
            # print(np.stack([target.numpy().flatten(),predi.flatten(),pred.flatten()],axis=1))
            bool_ls = pred.eq(target.view_as(pred)).view_as(target)
            bl = torch.where(bool_ls == True)[0]
            predictions.append(
                (torch.index_select(data, 0, bl),
                torch.index_select(target, 0, bl),
                torch.index_select(pred.view_as(target), 0, bl))
            )

def extract_images(predictions, plot_sample_count=20):
    shortlisted = []
    list_idx = torch.randint(0, len(predictions), (1,))[0]
    classes = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck",]
    smp_per_cls = np.floor(plot_sample_count/len(classes)).astype(int)

    # Create a dictionary to store indices for each class
    class_indices = {i: [] for i in range(len(classes))}

    # Collect indices of images by their predicted class
    for list_idx in range(len(predictions)):
        for batch_idx in range(predictions[list_idx][0].shape[0]):
            actual = predictions[list_idx][1][batch_idx].cpu().numpy()    # Actual class label
            predicted = predictions[list_idx][2][batch_idx].cpu().numpy() # Predicted class label
            if actual == predicted:  # Only consider images where the prediction is correct
                class_indices[int(actual)].append((predictions[list_idx][0][batch_idx], actual, predicted))
    # print([len(class_indices[k]) for k in class_indices.keys()])

    # Now, for each class, select at least "plot_sample_count" samples
    for class_id in range(len(classes)):  # Iterate over each class
        class_samples = class_indices[class_id]
        
        if len(class_samples) >= smp_per_cls:  # If we have enough samples, take "plot_sample_count" samples
            selected_samples = np.random.choice(len(class_samples), smp_per_cls, replace=False)
            
            for idx in selected_samples:
                image, actual, predicted = class_samples[idx]
                npimg = image.cpu().numpy()
                nptimg = np.transpose(npimg, (1, 2, 0))  # Convert to HxWxC format
                inverse_normalize(torch.Tensor(nptimg))
                shortlisted.append((nptimg, classes[actual], classes[predicted], actual, predicted))
        
        elif len(class_samples) > 0:
                image, actual, predicted = class_samples[0]
                npimg = image.cpu().numpy()
                nptimg = np.transpose(npimg, (1, 2, 0))  # Convert to HxWxC format
                inverse_normalize(torch.Tensor(nptimg))
                shortlisted.append((nptimg, classes[actual], classes[predicted], actual, predicted))

    return shortlisted


# [870, 928, 789, 680, 815, 787, 888, 872, 920, 907]

if __name__ == "__main__":
    # Plot the grad-Cam of ResNet18 model
    predictions = []

    net = ResNet18()
    weights = torch.load("saved/BestResNet18_k1_cifar_net_cpu.pth", map_location=torch.device("cpu"))
    net.load_state_dict(weights, strict=False)
    test_model(net, "cpu", testloader)

    plot_sample = 20        
    shortlisted_images = extract_images(predictions, plot_sample_count=plot_sample)
    target_layers = [net.layer3[-1]] # ResNet
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)

    fig = plt.figure(figsize=(12, 9))
    for i in range(len(shortlisted_images)):
        # 10 is the total number of classes
        a = fig.add_subplot(int(np.ceil(plot_sample/4.0)), 4, i+1)
        ip_img = shortlisted_images[i][0]
        input_tensor = torch.Tensor(np.transpose(ip_img, (2, 0, 1))).unsqueeze(dim=0)
        targets = [ClassifierOutputTarget(int(shortlisted_images[i][3]))]
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor,targets=targets, aug_smooth=True)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(ip_img, grayscale_cam, use_rgb=True)
        plt.imshow(visualization)

        a.axis("off")
        title = f"Actual: {shortlisted_images[i][1]} | Predicted: {shortlisted_images[i][2]}"
        a.set_title(title, fontsize=10)
    plt.savefig(str(f"{sv_fl}/grad_cam_features.png"),bbox_inches="tight",dpi=200)
    plt.close()

    # Plot the reconstruction
    batch_size = 64
    test_transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Extract the weights key from the dictionary first
    net2 = ConvAutoencoder()
    weights = torch.load("saved/best_ae_model.pth", map_location=torch.device("cpu"))
    net2.load_state_dict(weights["model_state_dict"],strict=False) # weights are loaded in place, does not require assignment
    net2.eval()

    for data, _ in testloader:
        images = data.to("cpu")
        output = net2(images)

    def display_images(original, decoded, count = 6): 
        n = count
        plt.figure(figsize=(12, 4))

        for i in range(n):
            # show original input image
            ax = plt.subplot(2, n, i+1)
            ax.imshow(np.transpose(original[1*i].cpu().numpy(), (1, 2, 0)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if i==0:
                ax.set_ylabel("input")

            # display decoded image
            ax = plt.subplot(2, n, i +1 + n)
            ax.imshow(np.transpose(decoded[1*i].detach().cpu().numpy(), (1, 2, 0)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if i==0:
                ax.set_ylabel("predicted")
        plt.savefig(f"{sv_fl}/autoencoder_images.png",bbox_inches="tight",dpi=200)
        plt.close()
    
    display_images(images, output)


    # Extract features from saved pytorch model - https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook


    net3 = copy.deepcopy(net2)
    net3.fc2.register_forward_hook(get_activation("fc2"))
    x = torch.randn(1, 25)
    output = net3(x)
    print(activation["fc2"])