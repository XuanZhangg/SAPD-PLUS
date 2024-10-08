from ctypes.wintypes import MAX_PATH
import torch
import numpy as np
import copy
from Optimization_Method import projection_simplex_sort as pj
import pickle
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)
from model_CIFAR10 import Model

#the main of SAPDx
def SAPDx(train_set,data_name,
        lr_x = 0.004,lr_y = 0.00001,theta0 = 0.8, rho = 1, eta = 1, theta_rate = 1.001,
        mu_y = 0.1, mu_x = 0.1, 
        b = 3000, b_1 = 200, q = 200, m = 10,
        sim_time = 1, max_epoch = 100, epoch_numer = 18000, 
        is_show_result = False, is_save_data = False, device = 'cuda'):  
    #initialize the result
    record_SAPDx = []
    acc = []
    #mu_y is the regulirazer for y
    start_model = Model(mu_y=mu_y,mu_x=mu_x).to(device)

    for s in range(sim_time):
        #initialize for this simulation
        epoch_SAPDx = []
        record_SAPDx_sub = []
        theta = theta0
        epoch,sample_comlexity = 0,0

        #load the start model
        test1 = Model(mu_y=mu_y,mu_x=mu_x).to(device)
        test1.load_state_dict(copy.deepcopy(start_model.state_dict()))
        epoch_SAPDx.append(0)
        record_SAPDx_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())

        #initilize the output
        output = []
        for param in test1.parameters():
            output.append(param)

        #outer loop
        while 1:
            #inner loop
            # theta = min(theta*1.01,0.9)
            theta = min(theta*theta_rate,0.9)
            N = int(np.log(256)/np.log(1/theta)+1)#:inner loop iteration number
            #lr_y = (1-rho)/(rho*mu_y)
            # lr_y = 0.00001
            # lr_x = 0.004
            #lr_y = 0.1

            Z0 = output #:Z0 is the inexact proximal center, which is the ouput of the last inner loop
            #initialize the proximal center
            test1.Z0 = []
            for param in output:
                test1.Z0.append(param.data)

            pre_oracles_dual = []
            iter = 0 #:the iteration number for the inner loop
            batch_start = 0 #:the start index of batch
            data_loader_dumb = torch.randperm(len(train_set)).to(device)

            while 1:   
                #re-initial the output as 0 vectors
                output = []
                for param in test1.parameters():
                    output.append(torch.zeros_like(param))

                #generate the batch select data by batch index
                if batch_start+b < len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                    sample_comlexity += b
                else:
                    #drop the incomplete data if they can not form a full batch
                    data_loader_dumb = torch.randperm(len(train_set)).to(device)
                    batch_start = 0
                    continue

                data = torch.index_select(train_set.data,0,index=batch_index) #:unseueeze is to make [64,28,28] to [64,1,28,28]
                target = torch.index_select(train_set.targets,0,index=batch_index)

                #compute the dual gradient of x_k,y_k
                test1.zero_grad()
                test1.obj(test1.forward(data),target).backward()
                oracles_dual = []
                for name,param in test1.named_parameters():
                    oracles_dual.append(param.grad) #:although we don't need primal here, but to use zip properly later, we need to have the same shape

                #update dual variables to x_k+1
                if not pre_oracles_dual:
                    pre_oracles_dual = copy.deepcopy(oracles_dual) #:only for the first step, pre_oracles_dual == oracles_dual
                for (name,param),grad,pre_grad in zip(test1.named_parameters(),oracles_dual,pre_oracles_dual):
                    if name != 'variable_y':
                        momentum = grad - pre_grad
                        param.data = param.data + lr_x*(grad.data + theta*momentum.data)
                #record the current dual gradeint of at point x_k+1,y_k before updating, pre_oracles_dual will be x_k-1,y_k-1 in the next iteration 
                pre_oracles_dual = copy.deepcopy(oracles_dual)

                #compute the primal gradient of x_k+1,y_k only for primal variable
                test1.zero_grad()
                test1.obj(test1.forward(data),target).backward()
                oracles_primal = [] #:oracles_dual is vt
                for name,param in test1.named_parameters():
                    oracles_primal.append(param.grad) #:although we don't need dual here, but to use zip properly later, we need to have the same shape

                #update primal variables to y_k+1
                for (name,param),grad in zip(test1.named_parameters(),oracles_primal):
                    if name == 'variable_y':
                        projection_center =  param - lr_y*grad 
                        param.data = torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device)
                        break
                
                #record the output of each inner loop and take the average
                for (name,param1), param2 in zip(test1.named_parameters(),output):
                    #param2.data = (iter*param2.data + param1.data)/(iter + 1)
                    param2.data =  param1.data

                iter += 1
                if iter == N:
                    break

            #update the model by the output of inner loop
            for param1,param2 in zip(test1.parameters(),output):
                param1.data = param2.data


            if sample_comlexity//epoch_numer>epoch:
                epoch = sample_comlexity//epoch_numer
                epoch_SAPDx.append(epoch)
                record_SAPDx_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
                acc.append(torch.sum(test1.predict(train_set.data)==train_set.targets)/len(train_set.data))
                if is_show_result:
                    print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SAPDx_sub[-1])
                
                #print('sample complexity is', sample_comlexity, ', epoch is', epoch,  ', loss is', record[-1])
                if epoch >= max_epoch+10:
                    break

        #save this simulation result
        print('')
        print('Simulation time ', s+1, ' is done.....')
        print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SAPDx_sub[-1])
        print('')
        record_SAPDx.append(record_SAPDx_sub)

        if is_save_data:
            file_name = './result_data/' +  data_name + '/SAPDx_' + 'theta=0.' + str(int(theta0*100))
            with open(file_name , "wb") as fp:  
                pickle.dump([record_SAPDx,epoch_SAPDx], fp)


#main of SAPD
def SAPD(train_set,data_name,
        lr_x = 0.004,lr_y = 0.00001,theta0 = 0.8, rho = 1, eta = 1, theta_rate = 1.001,
        mu_y = 0.1, mu_x = 0.1, 
        b = 3000, b_1 = 200, q = 200, m = 10,
        sim_time = 1, max_epoch = 100, epoch_numer = 18000, 
        is_show_result = False, is_save_data = False, device = 'cuda'):  
    #initialize the result
    record_SAPD = []
    acc = []
    #mu_y is the regulirazer for y
    start_model = Model(mu_y=mu_y,mu_x=mu_x).to(device)

    for s in range(sim_time):
        #initialize for this simulation
        epoch_SAPD = []
        record_SAPD_sub = []
        theta = theta0
        epoch,sample_comlexity = 0,0

        #load the start model
        test1 = Model(mu_y=mu_y,mu_x=mu_x).to(device)
        test1.load_state_dict(copy.deepcopy(start_model.state_dict()))
        epoch_SAPD.append(0)
        record_SAPD_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())

        #initilize the output
        output = []
        for param in test1.parameters():
            output.append(param)

        #outer loop
        while 1:
            #inner loop
            theta = min(theta*theta_rate,0.9)
            N = int(np.log(256)/np.log(1/theta)+1)#:inner loop iteration number
            #lr_y = (1-rho)/(rho*mu_y)
            # lr_y = 0.00001
            # lr_x = 0.004
            #lr_y = 0.1

            
            Z0 = copy.deepcopy(output) #:Z0 is the inexact proximal center, which is the ouput of the last inner loop
            pre_oracles_dual = []
            iter = 0 #:the iteration number for the inner loop
            batch_start = 0 #:the start index of batch
            data_loader_dumb = torch.randperm(len(train_set)).to(device)

            while 1:   
                #re-initial the output as 0 vectors
                output = []
                for param in test1.parameters():
                    output.append(torch.zeros_like(param))

                #generate the batch select data by batch index
                if batch_start+b < len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                    sample_comlexity += b
                else:
                    #drop the incomplete data if they can not form a full batch
                    data_loader_dumb = torch.randperm(len(train_set)).to(device)
                    batch_start = 0
                    continue

                data = torch.index_select(train_set.data,0,index=batch_index) #:unseueeze is to make [64,28,28] to [64,1,28,28]
                target = torch.index_select(train_set.targets,0,index=batch_index)

                #compute the dual gradient of x_k,y_k
                test1.zero_grad()
                test1.loss(test1.forward(data),target).backward()
                oracles_dual = []
                for name,param in test1.named_parameters():
                    oracles_dual.append(param.grad) #:although we don't need primal here, but to use zip properly later, we need to have the same shape

                #update dual variables to y_k+1
                if not pre_oracles_dual:
                    pre_oracles_dual = copy.deepcopy(oracles_dual) #:only for the first step, pre_oracles_dual == oracles_dual
                for (name,param),grad,pre_grad in zip(test1.named_parameters(),oracles_dual,pre_oracles_dual):
                    if name == 'variable_y':
                        momentum = grad - pre_grad
                        projection_center =  param + lr_y*(grad + theta*momentum)
                        param.data = torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device)
                        break #:only one variable_y variable, break for saving time
                #record the current dual gradeint of at point x_k,y_k before updating, pre_oracles_dual will be x_k-1,y_k-1 in the next iteration 
                pre_oracles_dual = copy.deepcopy(oracles_dual)

                #compute the primal gradient of x_k,y_k+1 only for primal variable
                test1.zero_grad()
                test1.loss(test1.forward(data),target).backward()
                oracles_primal = [] #:oracles_dual is vt
                for name,param in test1.named_parameters():
                    oracles_primal.append(param.grad) #:although we don't need dual here, but to use zip properly later, we need to have the same shape

                #update primal variables to x_k+1
                for (name,param),grad,x0 in zip(test1.named_parameters(),oracles_primal,Z0):
                    if name != 'variable_y':
                        param.data = param.data - lr_x * (grad.data + mu_x*(param.data-x0))
                
                #record the output of each inner loop and take the average
                for (name,param1), param2 in zip(test1.named_parameters(),output):
                    #param2.data = (iter*param2.data + param1.data)/(iter + 1)
                    param2.data =  param1.data

                iter += 1
                if iter == N:
                    break

            #update the model by the output of inner loop
            for param1,param2 in zip(test1.parameters(),output):
                param1.data = param2.data
            pre_test1_primal = Model(mu_y=mu_y,mu_x=mu_x).to(device)
            pre_test1_primal.load_state_dict(copy.deepcopy(test1.state_dict()))
            pre_test1_dual = Model(mu_y=mu_y,mu_x=mu_x).to(device)
            pre_test1_dual.load_state_dict(copy.deepcopy(test1.state_dict()))

            if sample_comlexity//epoch_numer>epoch:
                epoch = sample_comlexity//epoch_numer
                epoch_SAPD.append(epoch)
                record_SAPD_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
                acc.append(torch.sum(test1.predict(train_set.data)==train_set.targets)/len(train_set.data))
                if is_show_result:
                    print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SAPD_sub[-1])
                
                #print('sample complexity is', sample_comlexity, ', epoch is', epoch,  ', loss is', record[-1])
                if epoch >= max_epoch+10:
                    break

        #save this simulation result
        print('')
        print('Simulation time ', s+1, ' is done.....')
        print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SAPD_sub[-1])
        print('')
        record_SAPD.append(record_SAPD_sub)

        if is_save_data:
            file_name = './result_data/' +  data_name + '/SAPD_' + 'theta=0.' + str(int(theta0*100))
            with open(file_name , "wb") as fp:  
                pickle.dump([record_SAPD,epoch_SAPD], fp)


#main of SAPD_VR
def SAPD_VR(train_set,data_name,
        lr_x0 = 0.005,lr_y = 0.00001,theta0 = 0.8, rho = 1, eta = 1, theta_rate = 1.001,
        mu_y = 0.1, mu_x = 0.1, 
        b = 3000, b_1 = 200, q = 200, m = 10,
        sim_time = 1, max_epoch = 100, epoch_numer = 18000, 
        is_show_result = False, is_save_data = False, is_adaptive = False,device = 'cuda'):  
    # torch.manual_seed(2)
    # random.seed(2)
    # np.random.seed(2)
    record_SAPD_VR = []
    acc = []
    start_model = Model(mu_y=mu_y,mu_x=mu_x).to(device)

    s = 0
    while s <sim_time:
        #initialize for this simulation
        epoch_SAPD_VR = []
        record_SAPD_VR_sub = []
        theta = theta0
        epoch,sample_comlexity = 0,0
        min_loss = 10.0
        is_bad = False

        #load the start model
        test1 = Model(mu_y).to(device)
        test1.load_state_dict(copy.deepcopy(start_model.state_dict()))
        epoch_SAPD_VR.append(0)
        record_SAPD_VR_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())

        #initilize the output
        output = []
        pre = []
        for param in test1.parameters():
            output.append(param)

        #outer loop
        while 1:
            #inner loop
            theta = min(theta*1.01,0.9)
            N = int(np.log(256)/np.log(1/theta)+1)#:inner loop iteration number
            # lr_x = 0.005
            # lr_y = (1-theta)/(theta*mu_y)
            # lr_x = 0.004
            # lr_y = 0.00001
            #lr_x = max(lr_x0/(epoch//18+1),0.002)

            if is_adaptive:
                lr_x = max(lr_x0/(epoch//10+1),0.001)
            else:
                lr_x = lr_x0
            
            Z0 = output #:Z0 is the inexact proximal center, which is the ouput of the last inner loop
            pre_oracles_dual = []
            iter = 0 #:the iteration number for the inner loop
            batch_start = 0 #:the start index of batch
            data_loader_dumb = torch.randperm(len(train_set)).to(device)

            while 1:   
                #re-initial the output as 0 vectors
                output = []
                for param in test1.parameters():
                    output.append(torch.zeros_like(param))
                pre = copy.deepcopy(output)
                

                #generate the batch select data by batch index
                if np.mod(iter, q) == 0 and batch_start+b < len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                    sample_comlexity += b
                elif np.mod(iter, q) != 0 and batch_start+b_1 < len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b_1]
                    batch_start += b_1
                    sample_comlexity += b_1
                else:
                    #drop the incomplete data if they can not form a full batch
                    data_loader_dumb = torch.randperm(len(train_set)).to(device)
                    batch_start = 0
                    continue
                data = torch.index_select(train_set.data,0,index=batch_index) #:unseueeze is to make [64,28,28] to [64,1,28,28]
                target = torch.index_select(train_set.targets,0,index=batch_index)

                #compute the dual gradient of x_k,y_k
                test1.zero_grad()
                test1.loss(test1.forward(data),target).backward()
                if np.mod(iter, q) != 0:
                    #compute the dual gradient of x_k-1,y_k-1
                    pre_test1_dual.zero_grad()
                    pre_test1_dual.loss(pre_test1_dual.forward(data),target).backward()

                    temp = []
                    for (name,param1),param2,grad in zip(test1.named_parameters(), pre_test1_dual.parameters(), oracles_dual): #the fisrt para is variable_y
                            temp.append(param1.grad - param2.grad + grad) #:although we don't need primal here, but to use zip properly later, we need to have the same shape
                    oracles_dual = temp
                else:
                    oracles_dual = []
                    for name,param in test1.named_parameters():
                        oracles_dual.append(param.grad) #:although we don't need primal here, but to use zip properly later, we need to have the same shape

                #record the current point x_k,y_k before updating, pre_test1_dual will be x_k-1,y_k-1 in the next iteration     
                pre_test1_dual = Model(mu_y).to(device)
                pre_test1_dual.load_state_dict(copy.deepcopy(test1.state_dict()))

                #update dual variables to y_k+1
                if not pre_oracles_dual:
                    pre_oracles_dual[:] = oracles_dual #:only for the first step, pre_oracles_dual == oracles_dual
                for (name,param),grad,pre_grad in zip(test1.named_parameters(),oracles_dual,pre_oracles_dual):
                    if name == 'variable_y':
                        momentum = grad - pre_grad
                        projection_center =  param + lr_y*(grad + theta*momentum)
                        param.data = torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device)
                        break #:only one variable_y variable, break for saving time

                #record the dual VR gradient  for next iteration, it is x_k-1,y_k-1 in this iteration
                pre_oracles_dual = copy.deepcopy(oracles_dual)

                #compute the primal gradient of x_k,y_k+1 only for primal variable
                test1.zero_grad()
                test1.loss(test1.forward(data),target).backward()
                if np.mod(iter, int(b)) != 0:
                    #compute the primal gradient of x_k-1,y_k
                    pre_test1_primal.zero_grad() 
                    pre_test1_primal.loss(pre_test1_primal.forward(data),target).backward()

                    temp = []
                    for (name,param1),param2,grad in zip(test1.named_parameters(), pre_test1_primal.parameters(), oracles_primal):
                        temp.append(param1.grad - param2.grad + grad) #:although we don't need dual here, but to use zip properly later, we need to have the same shape
                    oracles_primal = temp
                else:
                    oracles_primal = [] #:oracles_dual is vt
                    for name,param in test1.named_parameters():
                        oracles_primal.append(param.grad) #:although we don't need dual here, but to use zip properly later, we need to have the same shape

                #record the current point x_k,y_k+1 before updating, pre_test1_dual will be x_k-1,y_k in the next iteration  
                pre_test1_primal = Model(mu_y).to(device)
                pre_test1_primal.load_state_dict(copy.deepcopy(test1.state_dict()))

                #update primal variables to x_k+1
                for (name,param),grad,x0 in zip(test1.named_parameters(),oracles_primal,Z0):
                    if name != 'variable_y':
                        param.data = param.data - lr_x * (grad.data + mu_x*(param.data-x0))
                
                #record the output of each inner loop and take the average
                for (name,param1), param2,param3 in zip(test1.named_parameters(),output,pre):
                    param3.data = (iter*param1.data + param1.data)/(iter + 1)
                    param2.data = param1.data

                iter += 1
                if iter >= N:
                    break

            #update the model by the output of inner loop
            for param1,param2 in zip(test1.parameters(),output):
                param1.data = param2.data

            pre_test1_primal = Model(mu_y).to(device)
            pre_test1_primal.load_state_dict(copy.deepcopy(test1.state_dict()))
            pre_test1_dual = Model(mu_y).to(device)
            pre_test1_dual.load_state_dict(copy.deepcopy(test1.state_dict()))
            
            if sample_comlexity//epoch_numer>epoch:
                epoch = sample_comlexity//epoch_numer
                epoch_SAPD_VR.append(epoch)
                record_SAPD_VR_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
                acc.append(torch.sum(test1.predict(train_set.data)==train_set.targets)/len(train_set.data))
                
                if is_show_result:
                    print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SAPD_VR_sub[-1])
                #print('sample complexity is', sample_comlexity, ', epoch is', epoch,  ', loss is', record_SAPD_VR_sub[-1])
                
                min_loss = np.min([record_SAPD_VR_sub[-1],min_loss])
                if record_SAPD_VR_sub[-1]>min_loss+0.5:
                    is_bad = True
                    break

                if epoch >= max_epoch+2:
                    break

        if is_bad:
            print('...bad simluation, break to next...')
            continue

        #save this simulation result
        s+=1
        print('')
        print('Simulation time ', s+1, ' is done.....')
        print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SAPD_VR_sub[-1])
        print('')
        record_SAPD_VR.append(record_SAPD_VR_sub)

        if is_save_data:
            file_name = './result_data/' +  data_name + '/SAPD_VR_' + 'theta=0.' + str(int(theta0*100))
            with open(file_name , "wb") as fp:  
                pickle.dump([record_SAPD_VR,epoch_SAPD_VR], fp)



#main of SMDA
def SMDA(train_set,data_name,
        lr_x = 0.004,lr_y = 0.00001,theta0 = 0.8, rho = 1, eta = 1, alpha = 0.1,
        mu_y = 0.1, mu_x = 0.1, 
        b = 3000, b_1 = 200, q = 200, m = 10,
        sim_time = 1, max_epoch = 100, epoch_numer = 18000, 
        is_show_result = False, is_save_data = False, device = 'cuda'): 
    # #random seed
    # torch.manual_seed(2)
    # np.random.seed(2)
    #initialize the result
    record_SMDA = []
    acc = []
    start_model = Model(mu_y=mu_y,mu_x=mu_x).to(device)

    for s in range(sim_time):
        #initialize for this simulation
        epoch_SMDA = []
        record_SMDA_sub = []
        epoch,sample_comlexity = 0,0

        #load the start model
        test1 = Model(mu_y=mu_y,mu_x=mu_x).to(device)
        test1.load_state_dict(copy.deepcopy(start_model.state_dict()))
        epoch_SMDA.append(0)
        record_SMDA_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
        
        #M is to generate matrix H:x,G:y
        M = []
        for param in test1.parameters():
            M.append(torch.zeros_like(param))

        epoch,iter,batch_start,sample_comlexity = 0,0,0,0
        data_loader_dumb = torch.randperm(len(train_set)).to(device)

        while 1:
            #select data by batch index
            if batch_start+b < len(data_loader_dumb):
                batch_index = data_loader_dumb[batch_start:batch_start+b]
                batch_start += b
                sample_comlexity += b
            else:
                #drop the incomplete data if they can not form a full batch
                data_loader_dumb = torch.randperm(len(train_set)).to(device)
                batch_start = 0
                continue
            data = torch.index_select(train_set.data,0,index=batch_index) 
            target = torch.index_select(train_set.targets,0,index=batch_index)

            ##compute the gradient of current point
            test1.zero_grad()
            test1.loss(test1.forward(data),target).backward()

            #oracles is the [vt,wt]
            oracles = []
            for name,param in test1.named_parameters():
                oracles.append(param.grad)
            
            ##compute the generate matrix H:x,G:y
            for (name,param), m in zip(test1.named_parameters(),M):
                m.data = alpha*m.data + (1-alpha)* (param.grad.data * param.grad.data)

            #update variables
            for (name,param),grad,m in zip(test1.named_parameters(),oracles,M):
                if name == 'variable_y':
                    #project variable_y onto simplex
                    projection_center = param + lr_y*(grad/(torch.sqrt(m)+rho)) 
                    param.data += eta*(torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device) - param.data)
                    torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device)
                else:
                    param.data = param.data - lr_x*(grad.data/(torch.sqrt(m.data)+rho)) 
                    
            iter += 1

            
            if sample_comlexity//epoch_numer>epoch:
                epoch = sample_comlexity//epoch_numer
                epoch_SMDA.append(epoch)
                record_SMDA_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
                acc.append(torch.sum(test1.predict(train_set.data)==train_set.targets)/len(train_set.data))
                if is_show_result:
                    print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SMDA_sub[-1])
                if epoch >max_epoch+2:
                    break

        #save this simulation result
        print('')
        print('Simulation time ', s+1, ' is done.....')
        print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SMDA_sub[-1])
        print('')
        record_SMDA.append(record_SMDA_sub)

        if is_save_data:
            file_name = './result_data/' +  data_name + '/SMDA'
            with open(file_name , "wb") as fp:  
                pickle.dump([record_SMDA,epoch_SMDA], fp)


#main of SMDA_VR
def SMDA_VR(train_set,data_name,
        lr_x = 0.001,lr_y = 0.00001,theta0 = 0.8, rho = 1, eta = 1, alpha = 0.1,
        mu_y = 0.1, mu_x = 0.1, 
        b = 3000, b_1 = 200, q = 200, m = 10,
        sim_time = 1, max_epoch = 100, epoch_numer = 18000, 
        is_show_result = False, is_save_data = False, device = 'cuda'): 
    #random seed
    # torch.manual_seed(2)
    # np.random.seed(2)
    #initialize the result
    record_SMDA_VR = []
    acc = []
    start_model = Model(mu_y=mu_y,mu_x=mu_x).to(device)
    is_bad = False
    min_loss = 100

    for s in range(sim_time):
        #initialize for this simulation
        epoch_SMDA_VR = []
        record_SMDA_VR_sub = []
        epoch,sample_comlexity = 0,0

        #load the start model
        test1 = Model(mu_y=mu_y).to(device)
        test1.load_state_dict(copy.deepcopy(start_model.state_dict()))
        epoch_SMDA_VR.append(0)
        record_SMDA_VR_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
        
        #M is to generate matrix H:x,G:y
        M = []
        for param in test1.parameters():
            M.append(torch.zeros_like(param))

        epoch,iter,batch_start,sample_comlexity = 0,0,0,0
        data_loader_dumb = torch.randperm(len(train_set)).to(device)

        while 1:
            #select data by batch index
            if np.mod(iter, q) == 0 and batch_start+b < len(data_loader_dumb):
                batch_index = data_loader_dumb[batch_start:batch_start+b]
                batch_start += b
                sample_comlexity += b
            elif np.mod(iter, q) != 0 and batch_start+b_1 < len(data_loader_dumb):
                batch_index = data_loader_dumb[batch_start:batch_start+b_1]
                batch_start += b_1
                sample_comlexity += b_1
            else:
                #drop the incomplete data if they can not form a full batch
                data_loader_dumb = torch.randperm(len(train_set)).to(device)
                batch_start = 0
                continue
            data = torch.index_select(train_set.data,0,index=batch_index) 
            target = torch.index_select(train_set.targets,0,index=batch_index)

            ##compute the gradient of current point
            test1.zero_grad()
            test1.loss(test1.forward(data),target).backward()

            if np.mod(iter, q) != 0:
                ##compute the gradient of previous point
                pre_test1.zero_grad()
                pre_test1.loss(pre_test1.forward(data),target).backward()

                temp = []
                # print('before updating')
                # print(oracles[0][0])
                for param1,param2,param3 in zip(test1.parameters(), pre_test1.parameters(), oracles):
                    temp.append(param1.grad - param2.grad + param3)
                oracles = copy.deepcopy(temp)
                # print('updated')
                # print(oracles[0][0])
            else:
                #oracles it the [vt,wt]
                oracles = []
                for name,param in test1.named_parameters():
                    oracles.append(param.grad)
            
            ##compute the generate matrix H:x,G:y
            for (name,param), m in zip(test1.named_parameters(),M):
                m.data = alpha*m.data + (1-alpha)* (param.grad.data * param.grad.data)

            #record the current model before updating
            pre_test1 = Model(mu_y).to(device)
            pre_test1.load_state_dict(copy.deepcopy(test1.state_dict()))

            # ii = 0
            # for param in pre_test1.parameters():
            #     ii += 1
            #     if ii == 2:
            #         print('pre_test data')
            #         print(param.data.data[0][0][0])
            #         break

            #update variables
            for (name,param),grad,m in zip(test1.named_parameters(),oracles,M):
                if name == 'variable_y':
                    #project variable_y onto simplex
                    projection_center = param + lr_y*(grad/(torch.sqrt(m)+rho)) 
                    param.data += eta*(torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device) - param.data)
                    torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device)
                else:
                    param.data = param.data - lr_x*(grad.data/(torch.sqrt(m.data)+rho)) 
                    
            iter += 1

            # ii = 0
            # for param1,param2 in zip(pre_test1.parameters(),test1.parameters()):
            #     ii += 1
            #     if ii == 2:
            #         print('pre_test data')
            #         print(param1.data[0][0][0])
            #         print('updated data')
            #         print(param2.data[0][0][0])
            #         break

            
            if sample_comlexity//epoch_numer>epoch:
                epoch = sample_comlexity//epoch_numer
                epoch_SMDA_VR.append(epoch)
                record_SMDA_VR_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
                acc.append(torch.sum(test1.predict(train_set.data)==train_set.targets)/len(train_set.data))
                print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SMDA_VR_sub[-1])

                min_loss = np.min([record_SMDA_VR_sub[-1],min_loss])
                if record_SMDA_VR_sub[-1]>min_loss+0.5: #0.5 
                    is_bad = True
                    break

                if epoch >= max_epoch+2:
                    break
        
        if is_bad:
            print('...bad simluation, break to next...')
            continue


        #save this simulation result
        print('')
        print('Simulation time ', s+1, ' is done.....')
        print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SMDA_VR_sub[-1])
        print('')
        record_SMDA_VR.append(record_SMDA_VR_sub)

        if is_save_data:
            file_name = './result_data/' +  data_name + '/SMDA_VR'
            with open(file_name , "wb") as fp:  
                pickle.dump([record_SMDA_VR,epoch_SMDA_VR], fp)


#main of PASGDA
def PASGD(train_set,data_name,
        lr_x = 0.004,lr_y = 0.00001,theta0 = 0.8, rho = 1, eta = 1, alpha = 0.1,
        mu_y = 0.1, mu_x = 0.1, 
        b = 3000, b_1 = 200, q = 200, m = 90,
        sim_time = 1, max_epoch = 100, epoch_numer = 18000, 
        is_show_result = False, is_save_data = False, device = 'cuda'):
    #initialize the result
    record_PASGDA = []
    acc = []
    start_model = Model(mu_y).to(device)

    for s in range(sim_time):
        #initialize for this simulation
        epoch_PASGDA = []
        record_PASGDA_sub = []
        epoch,sample_comlexity = 0,0

        #load the start model
        test1 = Model(mu_y).to(device)
        test1.load_state_dict(copy.deepcopy(start_model.state_dict()))
        epoch_PASGDA.append(0)
        record_PASGDA_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
        
        epoch,iter,batch_start,sample_comlexity = 0,0,0,0
        data_loader_dumb = torch.randperm(len(train_set)).to(device)

        while 1:  
            #select data by batch index
            if batch_start+b < len(data_loader_dumb):
                batch_index = data_loader_dumb[batch_start:batch_start+b]
                batch_start += b
                sample_comlexity += b
            else:
                #drop the incomplete data if they can not form a full batch
                data_loader_dumb = torch.randperm(len(train_set)).to(device)
                batch_start = 0
                continue
            data = torch.index_select(train_set.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
            target = torch.index_select(train_set.targets,0,index=batch_index)

            #compute the gradient of x_k,y_k
            test1.zero_grad()
            test1.loss(test1.forward(data),target).backward()
            oracles = [] #:oracles it the [vt,wt]
            for name,param in test1.named_parameters():
                oracles.append(param.grad)

            #update primal variables
            for (name,param),grad in zip(test1.named_parameters(),oracles):
                if name != 'variable_y':
                    param.data = param.data - lr_x*grad.data 
            
            #compute the gradient of x_k,y_k+1
            test1.zero_grad()
            test1.loss(test1.forward(data),target).backward()
            for (name,param),grad in zip(test1.named_parameters(),oracles):
                if name == 'variable_y':
                    #project variable_y onto simplex
                    projection_center = lr_y*grad + param
                    param.data = torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device)  



            if sample_comlexity//epoch_numer>epoch:
                epoch = sample_comlexity//epoch_numer
                epoch_PASGDA.append(epoch)
                record_PASGDA_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
                acc.append(torch.sum(test1.predict(train_set.data)==train_set.targets)/len(train_set.data))
                if is_show_result:
                    print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_PASGDA_sub[-1])
                if epoch > max_epoch+2:
                    break

        #save this simulation result
        print('')
        print('Simulation time ', s+1, ' is done.....')
        print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_PASGDA_sub[-1])
        print('')
        record_PASGDA.append(record_PASGDA_sub)

        if is_save_data:
            file_name = './result_data/' +  data_name + '/PASGDA'
            with open(file_name , "wb") as fp:  
                pickle.dump([record_PASGDA,epoch_PASGDA], fp)


#main of SREDA
def SREDA(train_set,data_name,
        lr_x = 0.004,lr_y = 0.00001,theta0 = 0.8, rho = 1, eta = 1, alpha = 0.1,
        mu_y = 0.1, mu_x = 0.1, 
        b = 3000, b_1 = 200, q = 200, m = 90,
        sim_time = 1, max_epoch = 100, epoch_numer = 18000, 
        is_show_result = False, is_save_data = False, device = 'cuda'):
    #random seed
    #initialize the result
    record_SREDA = []
    acc = []
    start_model = Model(mu_y=mu_y,mu_x=mu_x).to(device)
    is_bad = False
    min_loss = 100

    for s in range(sim_time):
        #initialize for this simulation
        epoch_SREDA = []
        record_SREDA_sub = []
        epoch,sample_comlexity = 0,0

        #load the start model
        test1 = Model(mu_y=mu_y,mu_x=mu_x).to(device)
        test1.load_state_dict(copy.deepcopy(start_model.state_dict()))
        epoch_SREDA.append(0)
        record_SREDA_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
        
        epoch,iter,batch_start,sample_comlexity = 0,0,0,0
        data_loader_dumb = torch.randperm(len(train_set)).to(device)

        while 1:
            # select data by batch index
            if np.mod(iter, q) == 0 and batch_start+b < len(data_loader_dumb):
                batch_index = data_loader_dumb[batch_start:batch_start+b]
                batch_start += b
                sample_comlexity += b
                # Sample data
                data = torch.index_select(train_set.data,0,index=batch_index)
                target = torch.index_select(train_set.targets,0,index=batch_index)
                
                ## compute the gradient of current point
                test1.zero_grad()
                test1.loss(test1.forward(data),target).backward()
                
                ## use the gradient to be the oracles
                oracles = []
                for name,param in test1.named_parameters():
                    oracles.append(param.grad)
            elif np.mod(iter, q) != 0:
                # use the last oracles
                pass
            else:
                #drop the incomplete data if they can not form a full batch
                data_loader_dumb = torch.randperm(len(train_set)).to(device)
                batch_start = 0
                continue

            #record the current param before updating
            pre_test1 = Model(mu_y=mu_y,mu_x=mu_x).to(device)
            pre_test1.load_state_dict(copy.deepcopy(test1.state_dict()))
            
            #update x first
            for (name,param),grad in zip(test1.named_parameters(),oracles):
                if name != 'variable_y':
                    param.data = param.data - lr_x*grad.data

            # ii = 0
            # for param in test1.parameters():
            #     if ii == 1:
            #         print('before max y')
            #         print(param.grad.data[0][0][0])
            #     ii += 1

            #update y using ConcaveMaximizer
            #x_k,y_k is pre_test, x_k+1,y_k is test,oracles are u_k, v_k
            t = 0
            data_loader_dumb = torch.randperm(len(train_set)).to(device)
            batch_start = 0

            while 1:
                # draw b_1 samples
                if  batch_start+b_1 < len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b_1]
                    batch_start += b_1
                    sample_comlexity += b_1
                else:
                    data_loader_dumb = torch.randperm(len(train_set)).to(device)
                    batch_start = 0
                    continue

                data = torch.index_select(train_set.data,0,index=batch_index)
                target = torch.index_select(train_set.targets,0,index=batch_index)
                
                # compute grad pre_test1, y_k-1
                pre_test1.zero_grad()
                pre_test1.loss(pre_test1.forward(data),target).backward()

                # compute grad pre_test1, y_k
                test1.zero_grad()
                test1.loss(test1.forward(data),target).backward()
                
                # update oracles
                temp = []
                for param1,param2,param3 in zip(test1.parameters(), pre_test1.parameters(), oracles):
                    temp.append(param1.grad - param2.grad + param3)
                oracles = temp

                # save pre_test before updating
                pre_test1 = Model(mu_y=mu_y,mu_x=mu_x).to(device)
                pre_test1.load_state_dict(copy.deepcopy(test1.state_dict()))

                # update y, x remain unchanged
                for (name,param),grad in zip(test1.named_parameters(),oracles):
                    if name == 'variable_y':
                        projection_center = param + lr_y*grad
                        if t == 0:
                            param.data = projection_center.data
                        else:
                            param.data = torch.tensor(pj(projection_center.cpu().detach().numpy()),dtype=torch.float32).to(device)

                
                t += 1
                if t > m+1:
                    break
            # ii = 0
            # for param in test1.parameters():
            #     if ii == 1:
            #         print('after max y')
            #         print(param.grad.data[0][0][0])
            #     ii += 1

            #orcales will be used in the next outer loop
        #    
                    
            iter += 1

            
            if sample_comlexity//epoch_numer>epoch:
                epoch = sample_comlexity//epoch_numer
                epoch_SREDA.append(epoch)
                record_SREDA_sub.append(test1.testloss(test1.forward(train_set.data),train_set.targets).cpu().detach().numpy())
                acc.append(torch.sum(test1.predict(train_set.data)==train_set.targets)/len(train_set.data))
                if is_show_result:
                    print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SREDA_sub[-1])
                
                min_loss = np.min([record_SREDA_sub[-1],min_loss])
                if record_SREDA_sub[-1]>min_loss+0.5: #0.5 
                    is_bad = True
                    break
                
                if epoch >= max_epoch+5:
                    break
        
        if is_bad:
            print('...bad simluation, break to next...')
            continue

        #save this simulation result
        print('')
        print('Simulation time is', s+1, ' .....')
        print('sample complexity is', sample_comlexity, ', epoch is', epoch, ', acc is', acc[-1], ', loss is', record_SREDA_sub[-1])
        print('')
        record_SREDA.append(record_SREDA_sub)

        if is_save_data:
            file_name = './result_data/' +  data_name + '/SREDA'
            with open(file_name , "wb") as fp:  
                pickle.dump([record_SREDA,epoch_SREDA], fp)