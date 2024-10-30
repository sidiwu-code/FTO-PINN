import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from sklearn import gaussian_process as gp
from scipy import interpolate
torch.set_default_dtype(torch.float32)

def fori_loop(lower, upper, body_fun, init_val):
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val
 
f = lambda x: np.sin(np.pi * x)
g = lambda t: np.sin(np.pi * t/2)

# Advection solver 
def solve_CVC(key, gp_sample, Nx, Nt, m, P):
    # Solve u_t + a(x) * u_x = 0
    # Wendroff for a(x)=V(x) - min(V(x)+ + 1.0, u(x,0)=f(x), u(0,t)=g(t)  (f(0)=g(0))
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    
    N = gp_sample.shape[0]
    X = np.linspace(xmin, xmax, N)[:,None]

 
    V = interpolate.interp1d(X.flatten(), gp_sample, kind='cubic', copy=False, assume_sorted=True) 
    # V = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h

    # Compute advection velocity
    def v_fn(x):
        return  V(x) - V(x.reshape(-1,)).min() + 1.0
    
    v =  v_fn(x)

    # Initialize solution and apply initial & boundary conditions
    u = np.zeros([Nx, Nt])
    u[0,:] = g(t)
    u[:,0] = f(x)
    
 

    # Compute finite difference operators
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)
    K = np.eye(Nx - 1, k=0)      # 
    K_temp = np.eye(Nx - 1, k=0)
    Trans = np.eye(Nx - 1, k=-1) # 
    def body_fn_x(i, carry):
        K, K_temp = carry
        K_temp = (-k[:, None]) * (Trans @ K_temp)
        K += K_temp
        return K, K_temp
    K, _ = fori_loop(0, Nx-2, body_fn_x, (K, K_temp))
    D = np.diag(k) + np.eye(Nx - 1, k=-1)
    
    def body_fn_t(i, u):
        b = np.zeros(Nx - 1)
        b[0] =  g(i * dt) - k[0] * g((i + 1) * dt)
        u[1:, i + 1] = K @ (D @ u[1:, i] + b)
       
        return u
    
    ## 
    UU = fori_loop(0, Nt-1, body_fn_t, u)

    # Input sensor locations and measurements
    xx = np.linspace(xmin, xmax, m)
    u = v_fn(xx)
    # Output sensor locations and measurements
    idx = random.sample(range(0,Nx),P)
    idt = random.sample(range(0,Nt),P)
 
    y = np.concatenate([x[idx][:,None], t[idt][:,None]], axis = 1)
    s = UU[idx, idt].reshape(-1,1)
 
    return (x, t, UU), (u, y, s)

 

# Computational domain
xmin = 0.0
xmax = 1.0

tmin = 0.0
tmax = 1.0

m = 100
P = 100



def Generate_test_data(N_test, Nx, Nt, gp_sample):
    N = 512
    X = np.linspace(xmin, xmax, num=N )[:, None]
 
 
    V = interpolate.interp1d(X.flatten(), gp_sample[0], kind='cubic', copy=False, assume_sorted=True) 
    # Compute advection velocity
    def v_fn(x):
        return  V(x) - V(x.reshape(-1,)).min() + 1.0
    
    (x, t, UU), (u, y, s) = solve_CVC(1, gp_sample[0], Nx, Nt, m, P)
    XX, TT = np.meshgrid(x, t)
    u_test = np.tile(u, (Nx*Nt,1))
    y_test = np.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
    s_test = UU.T.reshape(-1,1)
    
 
    for i in range(1,N_test):
        V = interpolate.interp1d(X.flatten(), gp_sample[i], kind='cubic', copy=False, assume_sorted=True) 
        # Compute advection velocity
        def v_fn(x):
            return  V(x) - V(x.reshape(-1,)).min() + 1.0
        
        (x, t, UU),  (u, y, s)= solve_CVC(1, gp_sample[i], Nx, Nt, m, P)
        u_test_ = np.tile(u, (Nx*Nt,1))
        y_test_ = np.hstack([XX.flatten()[:,None], TT.flatten()[:,None]])
        s_test_ = UU.T.reshape(-1,1)     
        u_test, y_test, s_test =  np.vstack((u_test,u_test_)), np.vstack((y_test,y_test_)), np.vstack((s_test,s_test_))
    return u_test, y_test, s_test, gp_sample


def Get_trunk(model, out_layer):
    """
    model
    out_layer: str. ex: 't1linear6'
    """
    trunk_model = nn.Sequential() 
 
    for layer in  model.named_modules(): 
        if layer[0] == 'actv': 
            actv_name, actv = layer[0], layer[1]

        elif layer[0][:8] == 't1linear' and layer[0]!=out_layer:
            # print('hh',layer[0])
            trunk_model.add_module(layer[0],nn.Sequential(
                layer[1],
                nn.Tanh()
            ))
        elif layer[0]==out_layer:
            # print('kk',layer[0])
            trunk_model.add_module(layer[0], layer[1])     
        else:
            pass
    return trunk_model

class DeepONet(nn.Module):
    def __init__(self, sensor_num=m, m=50,actv=nn.Tanh()):
        super(DeepONet, self).__init__()
        self.actv=actv
        self.b1linear_input = nn.Linear(sensor_num,m)    
        self.b1linear1 =nn.Linear(m,m)
        self.b1linear2 = nn.Linear(m,m)
        self.b1linear3 =nn.Linear(m,m)
        self.b1linear4 = nn.Linear(m,m)
        self.b1linear5 =nn.Linear(m,m)
        self.b1linear6 = nn.Linear(m,m)

        self.t1linear_input = nn.Linear(2,m)    
        self.t1linear1 =nn.Linear(m,m)
        self.t1linear2 = nn.Linear(m,m)
        self.t1linear3 =nn.Linear(m,m)
        self.t1linear4 = nn.Linear(m,m)
        self.t1linear5 = nn.Linear(m,m)
        self.t1linear6 = nn.Linear(m,m)

        self.p = self.__init_params() 
      
    def forward(self, X):
        feature = X[0]
        y = self.actv(self.b1linear_input(feature))
        y = self.actv(self.b1linear2(self.actv(self.b1linear1(y))))
        y = self.actv(self.b1linear4(self.actv(self.b1linear3(y))))
        branch =  self.b1linear6(self.actv(self.b1linear5(y)))
        
        y = self.actv(self.t1linear_input(X[1]))
        y = self.actv(self.t1linear2(self.actv(self.t1linear1(y))))
        y = self.actv(self.t1linear4(self.actv(self.t1linear3(y))))
        truck = self.t1linear6(self.actv(self.t1linear5(y)))

        return torch.sum(branch*truck, dim=-1, keepdim=True) + self.p['bias']   

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params
 

def a_function_interpololation(sample_i):
    N = 512
    X = np.linspace(xmin, xmax, num=N )[:, None]
 
    V = interpolate.interp1d(X.flatten(), sample_i, kind='cubic', copy=False, assume_sorted=True) 
    # Compute advection velocity
    def v_fn(x):
        return  V(x) - V(x.reshape(-1,)).min() + 1.0   
    
    return v_fn

class FNN(nn.Module):
    '''Fully-connected neural network.
    Note that
    len(size) >= 2,
    [..., N1, -N2, ...] denotes a linear layer from dim N1 to N2 without bias,
    [..., N, 0] denotes an identity map (as output linear layer).
    '''
    def __init__(self, size, init_range=1, activation= nn.Tanh()):
        super(FNN, self).__init__()
        self.size = size
        self.act = activation       
        self.ms = self.__init_modules()
        self.init_range = init_range
        self.__initialize()
        
    def forward(self, x):
        for i in range(1, len(self.size) - 1):
            x = self.act(self.ms['LinM{}'.format(i)](x))
        return self.ms['LinM{}'.format(len(self.size) - 1)](x) if self.size[-1] != 0 else x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            if self.size[i] != 0:
                bias = True if self.size[i] > 0 else False
                modules['LinM{}'.format(i)] = nn.Linear(abs(self.size[i - 1]), abs(self.size[i]), bias)
        return modules
    
    def __initialize(self):
        for i in range(1, len(self.size)):
            if self.size[i] != 0:
                self.ms['LinM{}'.format(i)].weight.data.uniform_(-self.init_range, self.init_range) 
                if self.size[i] > 0:
                    self.ms['LinM{}'.format(i)].bias.data.uniform_(-self.init_range, self.init_range) 

 

def Grad(y,x):
    '''return tensor([dfdx,dfdy,dfdz])
    '''    
    dydx, = torch.autograd.grad(outputs=y,inputs=x,retain_graph=True,grad_outputs=torch.ones(y.size()).to(x.device) ,
                                create_graph=True,allow_unused=True)
    return dydx



def FTO_grad(net_output, x_xiu, frequencies):
    """
    n = number of test data
    d = 2
    m = DOF
    inputs: 
        net_downput: net(x) Rn*Rm
        x:                  Rn*Rd
    """
    ux = []
    ut = []
    for i in range(net_output.shape[1]):
        g = Grad(net_output[:,i], x_xiu).cpu().detach().numpy()      # grad(phi)_x
        ux.append( g[:,0:1])
        ut.append( g[:,1:])
    ux = np.hstack(ux)
    ut = np.hstack(ut)

    if len(frequencies)>1:
        ux_xiu = []
        ut_xiu = []
        num = int(ux.shape[0]/len(frequencies))
        for i, f in enumerate(frequencies):
            ux_xiu.append(f * ux[i*num : (i+1)*num, :])
            ut_xiu.append(f * ut[i*num : (i+1)*num, :])
        ux_xiu = np.hstack(ux_xiu)
        ut_xiu = np.hstack(ut_xiu)
        return ux_xiu, ut_xiu
    else:
        return ux, ut


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

def FTO_approximator_solution(params, x,  frequencies_list):
    """
    model: neural network
    x: input locations 
    frequence: [1, 2, 4, 6, 8]:   a list
    """
    input_x = []
 
    for f in frequencies_list:
        input_x.append(f*x.detach())
    input_x = torch.vstack(input_x).requires_grad_(True).to(x.device)  

    model_output = model(params, input_x)
 
    return model_output, input_x

 
def transfer_FTO_approximate_solution(out, frequencies):
    num = int(out.shape[0]/len(frequencies))
    out_xiu = []
    for i, _ in enumerate(frequencies):
        out_xiu.append(  out[i*num : (i+1)*num, :])
    out_xiu = np.hstack(out_xiu)
    
    return out_xiu


def Generate_train_data(NUM):
    x =  np.linspace(xmin, xmax, num = NUM )[:, None]  ## 500
    xx, tt = np.meshgrid(x,x)
    xx = xx.flatten()
    tt = tt.flatten()

 
   
    x_domain = np.hstack((np.random.rand( NUM*2,1),np.random.rand(NUM*2,1)))

    x_boundary = np.hstack(( np.zeros((NUM, 1)), np.random.rand(NUM,1)))
    x_initial = np.hstack((np.random.rand(NUM,1),  np.zeros((NUM, 1))))

 

    x_domain = torch.tensor(x_domain).requires_grad_(True)
    x_initial = torch.tensor(x_initial)
    x_boundary = torch.tensor(x_boundary)

    return x_domain, x_initial, x_boundary

import time
def FTO_solver(PP_trunk, frequencies, x_domain, x_boundary, x_initial, y_test, s_test, a):
    t0= time.time()
    u, input_x_xiu = FTO_approximator_solution(PP_trunk, x_domain,  frequencies)
    ux, ut =  FTO_grad(u, input_x_xiu, frequencies)    
    ## ut+a ux = 0 
    A_domain = ut +np.diag(a.reshape(-1,))@ux 
    A_boundary  = FTO_approximator_solution(PP_trunk, x_boundary, frequencies)[0].cpu().detach().numpy()      
    A_boundary = transfer_FTO_approximate_solution(A_boundary, frequencies)

    A_initial = FTO_approximator_solution(PP_trunk, x_initial, frequencies)[0].cpu().detach().numpy()     
    A_initial = transfer_FTO_approximate_solution(A_initial, frequencies)
    A = np.vstack((A_domain,    A_boundary,   A_initial))

    ## rhs
    b_domain = np.zeros((ut.shape[0],1))
    b_boundary = np.sin(np.pi * x_boundary.cpu().detach().numpy()[:,1:]/2)
    b_initial = np.sin(np.pi * x_initial.cpu().detach().numpy()[:,0:1])
    b1 = np.vstack((b_domain,     b_boundary,   b_initial))

  
    dd =  np.linalg.norm(A, axis= 1)  
    cc =  np.diag( 1/ dd) 
    B = cc@A
    b2 = cc@b1
    c=np.linalg.lstsq(B, b2, rcond=None)
    ## record time
    t_fto1 = time.time()-t0
    coefficient =  c[0]      
    
    UU = s_test.cpu().detach().numpy().reshape(129, 129)
    pre = FTO_approximator_solution(PP_trunk,  y_test , frequencies)[0].cpu().detach().numpy()
    pre = transfer_FTO_approximate_solution(pre, frequencies)
    UU_pre = (pre@coefficient).reshape(129, 129)
    RL2_error_FTO = np.sqrt(((UU_pre-UU)**2).sum()/((UU)**2).sum())
    fto_linfty=np.max(np.abs(UU_pre-UU))
    print('The  accuracy and time of FTO-1:',  np.array(RL2_error_FTO).mean(), '&' , t_fto1, A.shape)
    return UU_pre, RL2_error_FTO, fto_linfty



def ELM_solver(model_elm,  x_domain, x_boundary, x_initial, y_test, s_test, a):
    t0= time.time()
    u = model_elm( x_domain)
    ux, ut =  FTO_grad(u, x_domain, frequencies=[1])    
    ## ut+a ux = 0 
    A_domain = ut +np.diag(a.reshape(-1,))@ux 
    A_boundary = model_elm( x_boundary).cpu().detach().numpy()      
    A_initial = model_elm( x_initial).cpu().detach().numpy()      
    A = np.vstack((A_domain,   A_boundary,   A_initial))

    ## rhs
    b_domain = np.zeros((ut.shape[0],1))
    b_boundary = np.sin(np.pi * x_boundary.cpu().detach().numpy()[:,1:]/2)
    b_initial = np.sin(np.pi * x_initial.cpu().detach().numpy()[:,0:1])
    b1 = np.vstack((b_domain,     b_boundary,   b_initial))


 
    dd =  np.linalg.norm(A, axis= 1)  
    cc =  np.diag( 1/ dd) 
    B = cc@A
    b2 = cc@b1
    c=np.linalg.lstsq(B, b2, rcond=None)
    ## record time
    t_fto1 = time.time()-t0
    coefficient =  c[0]      
 
    UU = s_test.cpu().detach().numpy().reshape(129, 129)
    UU_pre = (model_elm(  y_test ).cpu().detach().numpy()@coefficient).reshape(129, 129)
    RL2_error_FTO = np.sqrt(((UU_pre-UU)**2).sum()/((UU)**2).sum())
    elm_linfty=np.max(np.abs(UU_pre-UU))
    print('The  accuracy and time of elm-1:',  np.array(RL2_error_FTO).mean(), '&' , t_fto1, A.shape)
    return UU_pre, RL2_error_FTO, elm_linfty

