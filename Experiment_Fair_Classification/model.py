import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, mu_y=0.1, mu_x =0.1):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.variable_y = nn.Parameter(torch.tensor([1/3,1/3,1/3]), requires_grad=True)
        # super(Model,self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.bn1 = nn.BatchNorm2d(6, affine=False)
        # torch.nn.init.xavier_uniform(self.conv1.weight)

        # self.conv2 = nn.Conv2d(6, 16, 5)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        # self.bn2 = nn.BatchNorm2d(16, affine=False)

        # self.pool = nn.MaxPool2d(2, 2)

        # self.fc1 = nn.Linear(256, 120)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # self.fc2 = nn.Linear(120, 84)
        # torch.nn.init.xavier_uniform(self.fc2.weight)
        # self.fc3 = nn.Linear(84, 3)
        # torch.nn.init.xavier_uniform(self.fc3.weight)
        # self.variable_y = nn.Parameter(torch.tensor([1/3,1/3,1/3]), requires_grad=True)
        
        self.mu_y = mu_y
        self.mu_x = mu_x
        self.Z0 = []


    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim = 0)
        return x
    
    def loss(self,input,target):
        target_transform = F.one_hot(target,3).float()
        size = len(input)

        return -torch.matmul(
                            torch.log(torch.clamp(input[range(size), target],min=10**(-12))),
                            torch.matmul(target_transform,self.variable_y)
                            )/size - self.mu_y/2*torch.sum(self.variable_y **2)

    def testloss(self,input,target):
        target_transform = F.one_hot(target,3).float()
        size = len(input)

        return -torch.matmul(
                            torch.log(torch.clamp(input[range(size), target],min=10**(-12))),
                            torch.matmul(target_transform,self.variable_y)
                            )/size 

    def primal_loss(self,input,idx,target):
        # need to write according to test loss function
        pass

# class Model(nn.Module):
#     def __init__(self, mu_y=0.1, mu_x =0.1):
#         super(Model,self).__init__()
#         torch.manual_seed(2)
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.bn1 = nn.BatchNorm2d(6, affine=False)
#         torch.nn.init.xavier_uniform(self.conv1.weight)

#         self.conv2 = nn.Conv2d(6, 16, 5)
#         torch.nn.init.xavier_uniform(self.conv2.weight)
#         self.bn2 = nn.BatchNorm2d(16, affine=False)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(256, 120)
#         torch.nn.init.xavier_uniform(self.fc1.weight)
#         self.fc2 = nn.Linear(120, 84)
#         torch.nn.init.xavier_uniform(self.fc2.weight)
#         self.fc3 = nn.Linear(84, 3)
#         torch.nn.init.xavier_uniform(self.fc3.weight)
#         self.variable_y = nn.Parameter(torch.tensor([1/3,1/3,1/3]), requires_grad=True)
        
#         self.mu_y = mu_y
#         self.mu_x = mu_x
#         self.Z0 = []


#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.softmax(x,dim = 0)
#         return x
    
#     def loss(self,input,target):
#         target_transform = F.one_hot(target,3).float()
#         size = len(input)

#         return -torch.matmul(
#                             torch.log(torch.clamp(input[range(size), target],min=10**(-12))),
#                             torch.matmul(target_transform,self.variable_y)
#                             )/size - self.mu_y/2*torch.sum((self.variable_y * self.variable_y))


#     def obj(self,input,target):#obj for SAPDx subproblem
#         #min_y max_x
#         target_transform = F.one_hot(target,3).float()
#         size = len(input)

#         temp = 0

#         for param1,x0 in zip(self.parameters(),self.Z0):
#             temp += self.mu_x/2*torch.sum((param1-x0)*(param1-x0))

#         return torch.matmul(
#                             torch.log(torch.clamp(input[range(size), target],min=10**(-12))),
#                             torch.matmul(target_transform,self.variable_y)
#                             )/size + self.mu_y/2*torch.sum((self.variable_y * self.variable_y)) - temp


#     def predict(self, x):
#         return self.forward(x).argmax(dim=1)
