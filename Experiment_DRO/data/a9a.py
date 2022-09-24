
import torch
import pandas as pd
import pickle
from dataclass import Creatdata


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

X = pd.read_csv('./data/a9a/a9a.txt', sep="\t", header=None)

data = torch.tensor([[0 for _ in range(123)] for _ in range(len(X[0]))], dtype=torch.float32)
targets = torch.tensor([0 for _ in range(len(X[0]))], dtype=torch.int64)
for i in range(len(X[0])):
    item = X[0][i].split(' ')
    temp = torch.zeros(123, dtype=torch.float32)

    for char in item[1:]:
        ele = char.split(':')
        if len(ele) == 2:
            idx = int(ele[0])
            value = float(ele[1])
            temp[idx-1] = value
    
    data[i] = temp
    targets[i] = int(item[0])


train_set =  Creatdata(data,targets)

file_name = './data/' + data_name + '/' + data_name
with open(file_name , "wb") as fp:  
    pickle.dump(train_set, fp)