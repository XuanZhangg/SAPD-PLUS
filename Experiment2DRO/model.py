import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, data_size, lambda2=1e-3, mu_y=0.1, mu_x =0.1,alpha =10,device = 'cuda'):
        super(Model,self).__init__()
        self.d, self.n = data_size
        self.w = nn.Parameter(torch.zeros(self.d,1),requires_grad=True)
        self.variable_y = nn.Parameter(1/self.n*torch.torch.ones(self.n,), requires_grad=True)
        self.w = torch.nn.init.xavier_uniform(self.w)

        self.mu_y = mu_y
        self.mu_x = mu_x
        self.lambda2 = lambda2
        self.alpha = alpha
        self.Z0 = []
        self.device = device


    def forward(self, x):
        x = torch.matmul(x, self.w)
        return x
    
    def loss(self,input,idx,target):
        size = len(target)
        
        #regularizer part
        regularizer_x =  self.lambda2*torch.sum(self.alpha*self.w**2/(1 + self.alpha*self.w**2))
        regularizer_y =  1/2*1/(self.n)**2 * torch.sum((self.n*self.variable_y - 1)**2)

        #loss part
        bax = target.unsqueeze(1) *input #:ba is the log(1 + exp(-bax))
        logistic_loss = torch.zeros_like(bax, dtype = torch.float32)
        #case1:
        logistic_loss[bax <= -100.0] = -bax[bax <= -100.0]
        #case2:
        logistic_loss[bax > -100.0] = torch.log(1+torch.clamp(torch.exp(-bax[bax > -100.0]), min = 1e-12))
        weight_y = torch.index_select(self.variable_y,0,index=idx)

        return 1/size*torch.sum(logistic_loss*weight_y) + regularizer_x - regularizer_y
  
    
    def testloss(self,input,idx,target):
        size = len(target)
        
        #regularizer part
        regularizer_x =  self.mu_x*torch.sum(self.alpha*self.w**2/(1 + self.alpha*self.w**2))
        regularizer_y =  1/2*1/(self.n)**2 * torch.sum((self.n*self.variable_y - 1)**2)

        #loss part
        bax = target.unsqueeze(1) *input #:ba is the log(1 + exp(-bax))
        logistic_loss = torch.zeros_like(bax, dtype = torch.float32)
        #case1:
        logistic_loss[bax <= -100.0] = -bax[bax <= -100.0]
        #case2:
        logistic_loss[bax > -100.0] = torch.log(1+torch.clamp(torch.exp(-bax[bax > -100.0]), min = 1e-12))
        weight_y = torch.index_select(self.variable_y,0,index=idx)
  
        return 1/size*torch.sum(logistic_loss*weight_y) 

    def predict(self, x):
        judge = self.forward(x)>=0
        temp = torch.ones_like(judge, dtype = torch.int64)
        temp[judge == False] = torch.tensor(-1,dtype = torch.int64)
        return torch.flatten(temp)

