import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)
train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                             transform = transforms.Compose([
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
                             )

#Limit the data to digits 0,2,3, convert data type to float32, make shape of [28,28] to [1,28,28]
#classes = ('plane', 'car', 'bird', 'cat',
#          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
t1,t2,t3 = 0,4,9
train_set.data = torch.tensor(train_set.data,dtype=torch.float32)
train_set.targets = torch.tensor(train_set.targets,dtype=torch.int64)

idx = np.where((train_set.targets == t1) | (train_set.targets == t2) | (train_set.targets == t3))
train_set.data = train_set.data[idx]
train_set.data=train_set.data.permute(0, 3, 1, 2)
print(train_set.data.shape)
train_set.targets = train_set.targets[idx]


#change the targets to use onehot encoding
idx1 = train_set.targets == t1
idx2 = train_set.targets == t2
idx3 = train_set.targets == t3
train_set.targets[idx1] = 0
train_set.targets[idx2] = 1
train_set.targets[idx3] = 2

print('Examples of train set:') 
print(train_set.targets[0:10])


train_set.data = train_set.data.to(device)
train_set.targets = train_set.targets.to(device)