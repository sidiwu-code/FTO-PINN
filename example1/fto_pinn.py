# %%
from sklearn import gaussian_process as gp
import torch
import torch.nn as nn
import numpy as np
from Utils import  a_function_interpololation, FTO_approximator_solution, Generate_train_data, FTO_grad, FTO_solver
import os , time
data_place = './data_sample/'
if not os.path.isdir(data_place): os.makedirs(data_place)


# %%
 
N_test = 1 
length_scale = 0.2
device = 'cuda'
f_num = 7

### download gps
gp_sample = np.loadtxt(data_place+'length_scale_'+str(int(length_scale*100))+'.txt').reshape(1,-1)

 
Nx = 513
Nt = Nx
 
s_test =  np.loadtxt(data_place+'label_of_length_scale_'+str(int(length_scale*100))+'.txt').reshape(Nx, Nt)[::4,::4].reshape(-1,1)  
xx =  np.linspace(0, 1, 129)
tt =  np.linspace(0, 1, 129)
X, T = np.meshgrid(xx, tt)
y_test = np.hstack((X.reshape(-1,1), T.reshape(-1,1)))  ## test data points
a_function = a_function_interpololation(gp_sample[0])
u_test = np.tile(a_function(np.linspace(0,1,100)), (s_test.shape[0], 1)) ## sensor value

# %%
def load_model_parameters( mark,  params_place='./model_param_numpy/', device =device):
    tensor_param1 = []
    for  i in range(6):
        w = torch.tensor(np.loadtxt(params_place+mark+'_w_{}'.format(i))).float().to(device)
        b = torch.tensor(np.loadtxt(params_place+mark+'_b_{}'.format(i))).float().to(device)
        tensor_param1.append((w, b))

    U1, b1  =  torch.tensor(np.loadtxt(params_place + mark+'_U1')).float().to(device) , torch.tensor(np.loadtxt(params_place + mark+'_b1')).float().to(device)
    U2, b2  =  torch.tensor(np.loadtxt(params_place + mark+'_U2')).float().to(device), torch.tensor(np.loadtxt(params_place + mark+'_b2')).float().to(device)
    return tensor_param1, U1, b1, U2, b2
PP_trunk = load_model_parameters(mark='t')
PP_branch = load_model_parameters(mark='b')


def model(params, inputs):
    params, U1, b1, U2, b2 = params
    U = nn.Tanh()(inputs@U1+ b1)
    V = nn.Tanh()(inputs@U2 + b2)
    for W, b in params[:-1]:
        outputs =  nn.Tanh()(inputs@W + b)
        inputs = torch.multiply(outputs, U) + torch.multiply(1 - outputs, V) 
    W, b = params[-1]
    outputs = inputs@W + b
    
    return outputs


 
u_test, y_test, s_test =  torch.tensor(u_test).float().to(device) ,torch.tensor(y_test).float().to(device), torch.tensor(s_test).float().to(device)
t0 = time.time()


NUM = 2000
x_domain, x_initial, x_boundary = Generate_train_data(NUM)
x_domain, x_initial, x_boundary = x_domain.float().to(device), x_initial.float().to(device), x_boundary.float().to(device)

### test frequency  
tensor_param, U1, b1, U2, b2 =  PP_trunk
cc  = torch.abs(x_domain@U1)
U1_k_max = torch.min((3-b1)/cc)
cc  = torch.abs(x_domain@U2)
U2_k_max = torch.min((3-b2)/cc)

KKK = int(min((U1_k_max, U2_k_max)))
a = a_function(x_domain.cpu().detach().numpy()[:,0])

 
frequencies = [1]
UU_pre,_,_ = FTO_solver(PP_trunk, frequencies, x_domain, x_boundary, x_initial, y_test, s_test, a)

 
frequencies = list(np.linspace(0.1, KKK, f_num-1))+[1] 
UU_pre_R,_,_ = FTO_solver(PP_trunk, frequencies, x_domain, x_boundary, x_initial, y_test, s_test, a)
 
 


