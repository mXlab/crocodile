#building the network
##COMMENTER ÉTAPE ICI

import torch.nn as nn
import torch.nn.functional as F

#building the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d() 
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10 ) 
    
#method that compute the data through the defined layers
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=16, subsampling=False):
        super(ResBlock, self).__init__()

        # If subsampling is true then downsample input by a factor 2
        stride = 1
        if subsampling:
            stride = 2

        network = [nn.BatchNorm1d(n_in), nn.ReLU(inplace=True), 
                   nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride),
                   nn.BatchNorm1d(n_out), nn.ReLU(inplace=True),
                   nn.Conv1d(n_out, n_out, kernel_size=kernel_size, stride=stride)]
        self.network = nn.Sequential(*network)

        # Create a functions to downsample input such that output and input have same dimension 
        self.subsampling = None  
        if subsampling or (n_in != n_out):
            self.subsampling = nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.network(x)
        if self.subsampling is not None:
            x = self.subsampling(x)

        return x + out


class ECGResNet(nn.Module):
    def __init__(self, n_in, num_classes, num_blocks=16, num_filters=32, kernel_size=16):
        super(ECGResNet, self).__init__()
        network = [nn.Conv1d(n_in, n_in, kernel_size=kernel_size)]
        for i in range(num_blocks):
            subsampling = False
            if (i+1) % 2 == 0:
                subsampling = True
            
            if (i+1) % 4 == 0:
                num_filters_new = num_filters*2
                network.append(ResBlock(num_filters, num_filters_new, subsampling=subsampling, kernel_size=kernel_size=))
                num_filters = num_filters_new
            else:
                network.append(ResBlock(num_filters, num_filters, subsampling=subsampling, kernel_size=kernel_size=))

        network.append(nn.AdaptiveAvgPool1d(1))
        self.network = nn.Sequential(*network)

        self.output_layer = nn.Linear(num_filters, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.network(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        return x

            



