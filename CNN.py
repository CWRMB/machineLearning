import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Adapted from https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # input is color
        # image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting,
        # stride is 1 , padding is 1 extract all edge features.
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # We double the feature maps for every conv
        # layer as in pratice it is really good.
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(4*4*64, 500) # I/p image size is 32*32, after 3 MaxPooling
        # layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10) # output nodes are 10 because our dataset have 10 different categories


    def forward(self, x):
        x = F.relu(self.conv1(x)) #Apply relu to each output of conv layer.
        x = F.max_pool2d(x, 2, 2)  # Max pooling layer with kernal of 2 and stride of 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)  # flatten our images to 1D to input it to the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Applying dropout b/t layers which exchange highest parameters. This is a good practice
        x = self.fc2(x)
        return x