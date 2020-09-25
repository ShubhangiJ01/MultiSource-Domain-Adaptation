import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3,12, kernel_size=4, stride=1, padding = 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12,12,kernel_size=4, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12,24, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(64896,128)
        self.bn1_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,64)
        self.bn2_fc = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self, x,reverse=False):
        #print(x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        #print(x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.shape)
        x = x.view(x.size(0), 64896)
	#x = x.view(x.size(0),864)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        # self.fc1 = nn.Linear(8192, 3072)
        # self.bn1_fc = nn.BatchNorm1d(3072)
        # self.fc2 = nn.Linear(3072, 2048)
        # self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(64,4)
        #x = nn.Linear(64,4)
        #F.softmax(x,1)
        #F.softmax(self.fc3,1)
        
        #self.bn_fc3 = nn.BatchNorm1d(4)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        x = self.fc3(x)
	#x = torch.nn.Softmax(x)
        # if reverse:
        #     x = grad_reverse(x, self.lambd)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        #x = self.fc3(x)
        x = F.softmax(x,1)
	#x = F.softmax(x,1)
        return x
