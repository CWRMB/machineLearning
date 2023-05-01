import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
import neptune
from torch.utils.data import random_split
from torchvision.models import ResNet101_Weights
from timm import models

from CNN import *

PATH = './cifar_net.pth'

rate_learning = 0.001

run = neptune.init_run(
    name="ResNet18(fc=256) & DropOut & 224Image",
    project="vidarlab/CIFA10Training",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9uZXctdWkubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzhhZjM5OS1kZjdjLTQ3MzAtODcyMS0yN2JiMWQyNDhhMGYifQ==",
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load image train & test data
transform = transforms.Compose(
     [transforms.RandomHorizontalFlip(),
      transforms.RandomAutocontrast(),
      transforms.RandomGrayscale(),
      transforms.Resize(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

batch_size = 256
epochs = 32

params = {
    "learning_rate": rate_learning,
    "optimizer": "SGD",
    "batch_size":batch_size,
    "epochs": epochs
}
run["parameters"] = params


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

validset = testset
# setup our validation loader
validloader = torch.utils.data.DataLoader(validset, batch_size)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(net):
    # Define loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), PATH)


def train_gpu(net):
    # Define loss function & optimizer
    criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100,
                                    reduce=None, reduction='mean', label_smoothing=0.1)
    # optimizer = optim.RMSprop(net.parameters(), lr=rate_learning, alpha=0.99,
    #                           eps=1e-08,weight_decay=0.001, momentum=0, centered=False)
    optimizer = optim.SGD(net.parameters(), lr=rate_learning, momentum=0.9)

    #min_valid_loss = np.inf
    min_valid_acc = 0

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_correct = 0
        val_correct = 0
        total = 0
        val_total = 0

        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate loss and number of correct
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_correct += (preds == labels).sum().item()

        # Calculate our validation loss and accuracy
        with torch.no_grad():
            valid_loss = 0.0
            net.eval()
            for j, valid_data in enumerate(validloader,0):
                valid_inputs, valid_labels = valid_data[0].to(device), valid_data[1].to(device)
                valid_outputs = net(valid_inputs)
                valid_loss += criterion(valid_outputs, valid_labels).item()
                _, val_preds = torch.max(valid_outputs.data, 1)
                val_total += valid_labels.size(0)
                val_correct += (val_preds == valid_labels).sum().item()

            valid_loss /= len(validloader)

        # Calculate the training loss and accuracy
        train_loss = running_loss / len(trainloader)
        accu = 100.*running_correct/total
        val_accu = 100.*val_correct/val_total

        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss:.6f}'
              f' \t\t Validation Loss: {valid_loss:.6f}' 
              f'\t\t Acc%: {accu:.3f}'f'\t\t Valid_Acc%: {val_accu:.3f}')

        run["train/valid_loss"].append(valid_loss)
        run["train/loss"].append(train_loss)
        run["train/accuracy"].append(accu)
        run["train/val_accuracy"].append(val_accu)

        # if valid_loss < min_valid_loss:
        #     print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})'
        #           f' \t Saving The Model')
        #     min_valid_loss = valid_loss
        #     torch.save(net.state_dict(), PATH)
        if val_accu > min_valid_acc:
            print(f'Validation Accuracy increased({min_valid_acc:.6f}--->{val_accu:.6f})'
                  f' \t Saving The Model')
            min_valid_acc = val_accu
            torch.save(net.state_dict(), PATH)


    print('Finished Training')


def test(net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def test_gpu(net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def classAccuracy(net):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def main():
    #net = LeNet()
    # net = torchvision.models.resnet18(pretrained=True)
    #net = ResNetWithDropout(num_classes=10, p=0.5)
    #1 net = ResNetWithReducedFC(num_classes=10, p=0.5, fc_channels=256)
    #net = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)

    net = models.vit_tiny_patch16_224(pretrained=True)

    #net.load_state_dict(torch.load(PATH))

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    if torch.cuda.is_available():
        print('Switching to cuda device')
        net.to(device)
        train_gpu(net)
        net.load_state_dict(torch.load(PATH))
        test_gpu(net)
        classAccuracy(net)
    else:
        print('Training on cpu')
        train(net)
        test(net)
        #classAccuracy(net)
    run.stop()
    # Show images and ground-truth vs predicted
    #plotImage(net)


if __name__ == '__main__':
    main()
