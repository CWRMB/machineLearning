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

rate_learning = 0.001

run = neptune.init_run(
    name="ResNet18 & DropOut & Data Aug",
    project="vidarlab/CIFA10Training",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vbmV3LXVpLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9uZXctdWkubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzhhZjM5OS1kZjdjLTQ3MzAtODcyMS0yN2JiMWQyNDhhMGYifQ==",
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#writer = SummaryWriter('runs/resnet18')

# Load image train & test data
transform = transforms.Compose(
     [transforms.RandomHorizontalFlip(),
      transforms.RandomGrayscale(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

batch_size = 256

params = {
    "learning_rate": rate_learning,
    "optimizer": "RMSProp",
    "batch_size":batch_size,
    "epochs": 32
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=rate_learning, alpha=0.99,
                              eps=1e-08,weight_decay=0.0001, momentum=0, centered=False)
    min_valid_loss = np.inf

    for epoch in range(32):  # loop over the dataset multiple times
        running_loss = 0.0
        running_correct = 0
        total = 0
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
            train_loss = running_loss / len(trainloader)

            if i % 10 == 9:  # print every 10 mini-batch
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f} '
                      f'Acc: {100 * running_correct // total:.3f}')
                #train_loss = running_loss / 10
                running_loss = 0.0

                # Calculate our validation loss
                with torch.no_grad():
                    valid_loss = 0.0
                    net.eval()
                    for j, valid_data in enumerate(validloader,0):
                        valid_inputs, valid_labels = valid_data[0].to(device), valid_data[1].to(device)
                        valid_outputs = net(valid_inputs)
                        valid_loss = criterion(valid_outputs, valid_labels) * valid_inputs.size(0)

                    valid_loss /= len(validloader)

                # Calculate the training loss
                print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss:.6f}'
                      f' \t\t Validation Loss: {valid_loss:.6f}')

                run["train/valid_loss"].append(valid_loss)
                run["train/loss"].append(train_loss)
                run["train/accuracy"].append(100 * running_correct // total)

                running_correct = 0

                if min_valid_loss > valid_loss:
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})'
                          f' \t Saving The Model')
                    min_valid_loss = valid_loss
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
    # net = torchvision.models.resnet18(pretrained=True)
    net = ResNetWithDropout(num_classes=10, p=0.5)
    #net.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))

    #net.load_state_dict(torch.load(PATH))

    # net.fc = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(net.fc.in_features, 10)
    # )

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
