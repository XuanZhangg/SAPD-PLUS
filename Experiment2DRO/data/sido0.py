
import copy
import torch
import pandas as pd
import pickle
from dataclass import Creatdata
import bz2
import scipy.io

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)


data = scipy.io.loadmat('./data/sido0/sido0_train.mat') 
data = torch.tensor(data['data'],dtype=torch.float32, device=device)


targets =  pd.read_csv('./data/sido0/sido0_train.targets', header=None)
targets = torch.tensor(targets.values.tolist(),dtype=torch.int64, device=device).squeeze(1)

train_set =  Creatdata(data,targets)
data_name = 'sido0'
file_name = './data/' + data_name + '/' + data_name
with open(file_name , "wb") as fp:  
    pickle.dump(train_set, fp)
