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

from CNN import *

PATH = './cifar_net.pth'

rate_learning = 0.0001

run = neptune.init_run(
    project="vidarlab/CIFA10Training",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9uZXctdWkubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzhhZjM5OS1kZjdjLTQ3MzAtODcyMS0yN2JiMWQyNDhhMGYifQ==",
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#writer = SummaryWriter('runs/resnet18')

# Load image train & test data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 256

params = {
    "learning_rate": rate_learning,
    "optimizer": "Adam",
    "batch_size":batch_size,
    "epochs": 24
}
run["parameters"] = params


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Split our data ratio for validation and training loss
trainset, validset = random_split(trainset, [0.833 * len(trainset), 0.166 * len(trainset)])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
# setup our validation loader
validloader = torch.utils.data.DataLoader(validset, batch_size)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# def plotImage(net):
#     # get some random training images
#     dataiter = iter(trainloader)
#     images, labels = next(dataiter)
#
#     # print images
#     imshow(torchvision.utils.make_grid(images))
#     print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
#
#     outputs = net(images)
#
#     _, predicted = torch.max(outputs, 1)
#
#     print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                                   for j in range(4)))


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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=rate_learning)


    for epoch in range(24):  # loop over the dataset multiple times

        running_loss = 0.0
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

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                #writer.add_scalar('training loss', running_loss / 50, epoch * len(trainloader) + i)
                # run["train/loss"].append(running_loss / len(testloader))
                running_loss = 0.0

        #Calculate our validation loss
        valid_loss = 0.0
        net.eval()
        for i, data in enumerate(validloader,0):
            inputs, labels = data[0].to(device), data[1].to(device)

            target = net(inputs)
            loss = criterion(target, labels)
            valid_loss = loss.item() * inputs.size(0)

            # if i % 10 == 9:
            #     print(f'[{epoch + 1}, {i + 1:5d}] valid_loss: {valid_loss / len(validloader):.3f}')
            #     run["train/valid_loss"].append(valid_loss / len(validloader))
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(trainloader)}'
            f' \t\t Validation Loss: {valid_loss / len(validloader)}')
        run["train/loss"].append(running_loss / len(testloader))
        run["train/valid_loss"].append(valid_loss / len(validloader))

    print('Finished Training')
    torch.save(net.state_dict(), PATH)

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
            #run["test/accuracy"].append(100*correct // total)

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
    net = torchvision.models.resnet18(pretrained=True)
    #net.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
    net.load_state_dict(torch.load(PATH))
    net.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(net.fc.in_features, 10)
    )

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    if torch.cuda.is_available():
        print('Switching to cuda device')
        net.to(device)
        train_gpu(net)
        test_gpu(net)
        classAccuracy(net)
    else:
        print('Training on cpu')
        train(net)
        test(net)
        #classAccuracy(net)
    run.stop()
    #dataiter = iter(trainloader)
    #writer.add_graph(net, next(dataiter))

    #writer.close()
    # Show images and ground-truth vs predicted
    #plotImage(net)


if __name__ == '__main__':
    main()
