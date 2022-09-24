
import torch
import torchvision
from torchvision import datasets, transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)
train_set = torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                             transform = transforms.Compose([
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
                             )

#Limit the data to digits 0,2,3, convert data type to float32, make shape of [28,28] to [1,28,28]
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot
idx = (train_set.targets == 0) | (train_set.targets == 5) | (train_set.targets == 9) 
train_set.targets = train_set.targets[idx]
train_set.data = train_set.data[idx].type(torch.float32).unsqueeze(1)

#change the targets to use onehot encoding
idx2 = train_set.targets == 5
idx3 = train_set.targets == 9
train_set.targets[idx2] = 1
train_set.targets[idx3] = 2

print('Examples of train set:') 
print(train_set.targets[0:10])
print(train_set.data.shape)

train_set.data = train_set.data.to(device)
train_set.targets = train_set.targets.to(device)