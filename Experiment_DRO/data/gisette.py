
import imp
import torch
import pandas as pd
import pickle
from dataclass import Creatdata
import bz2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)


data = torch.tensor([[0 for _ in range(5000)] for _ in range(6000)], dtype=torch.float32)
targets = torch.tensor([0 for _ in range(6000)], dtype=torch.int64)

i = 0
with bz2.open("./data/gisette/gisette.bz2", "rt") as bz_file:
    for item in bz_file:
        item = item.rstrip('\n').split(' ')
        temp = torch.zeros(5000, dtype=torch.float32)

        for char in item[1:]:
            ele = char.split(':')
            if len(ele) == 2:
                idx = int(ele[0])
                value = float(ele[1])
                temp[idx-1] = value
        
        data[i] = temp
        targets[i] = int(item[0])
        print(i)
        i += 1

train_set =  Creatdata(data,targets)

file_name = './data/' + data_name + '/' + data_name
with open(file_name , "wb") as fp:  
    pickle.dump(train_set, fp)