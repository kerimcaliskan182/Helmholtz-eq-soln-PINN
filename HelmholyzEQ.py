import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import qmc

##################################################################
########################Data Preparation##########################
##################################################################

# Number of boundary conditions
n_bc = 4

# Number of data points per boundary condition
n_data_per_bc = 20

# Generating Latin Hypercube Samples using QMC method
engine = qmc.LatinHypercube(d=1)

# Initializing a numpy array to store the data
data = np.zeros([n_bc, n_data_per_bc, 5])

# Setting the upper and lower bounds for the data
lim1 = 3.0 / 2
lim2 = 3

# Generating data for each boundary condition
for i, j in zip(range(n_bc), [0., 1., 0, 1.]):
    # Generating random points using Latin Hypercube Sampling
    points = (engine.random(n=n_data_per_bc)[:, 0] - 0.) * 1
    # Uncomment below to use linearly spaced points instead of QMC
    # points = np.linspace(0, +1, n_data_per_bc)
    
    # Assigning data based on boundary conditions
    if i < 2:
        data[i, :, 0] = j + 0.
        data[i, :, 1] = points - 0.
    else:
        data[i, :, 0] = points + 0.
        data[i, :, 1] = j - 0.

# Scaling the data to fit within specified bounds
data[:, :, 0] = 2 * lim1 * data[:, :, 0] - lim1
data[:, :, 1] = data[:, :, 1] * lim2



##################################################################
########################Boundary Conditions#######################
##################################################################

# Mode coefficients for boundary conditions
nn3 = 3   # Mode 3 coefficient
nn1 = 1   # Mode 1 coefficient
nn2 = 2   # Mode 2 coefficient

# Parameter for the equation
lam = 0.8

# Coefficients for boundary conditions
a3 = 1
a2 = 0
a1 = 1

# Calculating wave numbers for each mode
kk3 = np.sqrt(nn3**2 * np.pi**2 / lim2 / lim2 - lam**2)
kk1 = np.sqrt(nn1**2 * np.pi**2 / lim2 / lim2 - lam**2)
kk2 = np.sqrt(nn2**2 * np.pi**2 / lim2 / lim2 - lam**2)

# True solution function
def tru(x, y):      
    """
    Calculates the true solution at a given point (x, y).
    
    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        
    Returns:
        float: True solution value at (x, y)
    """
    tru = a3 * np.cos(nn3 * np.pi * x / lim2) * np.exp(-kk3 * y) + \
          a1 * np.cos(nn1 * np.pi * x / lim2) * np.exp(-kk1 * y) + \
          a2 * np.cos(nn2 * np.pi * x / lim2) * np.exp(-kk2 * y)
    return tru

# Derivative of true solution with respect to x
def truderx(x, y):      
    """
    Calculates the derivative of the true solution with respect to x at a given point (x, y).
    
    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        
    Returns:
        float: Derivative of the true solution with respect to x at (x, y)
    """
    truderx = -np.pi * nn3 / lim2 * a3 * np.sin(nn3 * np.pi * x / lim2) * np.exp(-kk3 * y) - \
              np.pi * nn1 / lim2 * a1 * np.sin(nn1 * np.pi * x / lim2) * np.exp(-kk1 * y) - \
              np.pi * nn2 / lim2 * a2 * np.sin(nn2 * np.pi * x / lim2) * np.exp(-kk2 * y)
    return truderx

# Derivative of true solution with respect to y
def trudery(x, y):      
    """
    Calculates the derivative of the true solution with respect to y at a given point (x, y).
    
    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        
    Returns:
        float: Derivative of the true solution with respect to y at (x, y)
    """
    trudery = -kk3 * a3 * np.cos(nn3 * np.pi * x / lim2) * np.exp(-kk3 * y) - \
              kk1 * a1 * np.cos(nn1 * np.pi * x / lim2) * np.exp(-kk1 * y) - \
              kk2 * a2 * np.cos(nn2 * np.pi * x / lim2) * np.exp(-kk2 * y)
    return trudery


##################################################################
########################Data Processing###########################
##################################################################

# Printing the y-values of the fourth boundary condition data
print(data[3,:,1])

# Processing boundary condition data
for j in range(0, n_data_per_bc):    
    # Boundary condition 1: Left boundary
    data[0, j, 2] = tru(-lim1, data[0, j, 1])
    data[0, j, 3] = truderx(-lim1, data[0, j, 1]) * 0
    data[0, j, 4] = trudery(-lim1, data[0, j, 1]) * 1
           
    # Boundary condition 2: Right boundary
    data[1, j, 2] = tru(lim1, data[1, j, 1])
    data[1, j, 3] = truderx(lim1, data[1, j, 1]) * 0
    data[1, j, 4] = trudery(lim1, data[1, j, 1]) * 1

    
for i in range(0, n_data_per_bc):
    # Boundary condition 3: Bottom boundary
    data[2, i, 2] = tru(data[2, i, 0], 0)
    data[2, i, 3] = truderx(data[2, i, 0], 0)
    data[2, i, 4] = trudery(data[2, i, 0], 0)

    # Boundary condition 4: Top boundary
    data[3, i, 2] = tru(data[3, i, 0], lim2)
    data[3, i, 3] = truderx(data[3, i, 0], lim2)
    data[3, i, 4] = trudery(data[3, i, 0], lim2)

# Reshaping data array for further processing
data = data.reshape(n_data_per_bc * n_bc, 5)

# Separating the solution data and associated derivatives
x_d, y_d, t_d, t_dx, t_dy = map(lambda x: np.expand_dims(x, axis=1), 
                                [data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]])

# Number of collocation points
Nc = 700

# Generating collocation points using Latin Hypercube Sampling in 2 dimensions
engine = qmc.LatinHypercube(d=2)
colloc = engine.random(n=Nc)
colloc = 1 * (colloc - 0)

# Scaling collocation points to fit within specified bounds
colloc[:, 0] = 2 * lim1 * colloc[:, 0] - lim1
colloc[:, 1] = lim2 * colloc[:, 1] 

# Separating x and y coordinates of collocation points
x_c, y_c = map(lambda x: np.expand_dims(x, axis=1), [colloc[:, 0], colloc[:, 1]])

# Converting variables to torch tensors for use in the model
x_c_tensor = torch.tensor(x_c, dtype=torch.float32)
y_c_tensor = torch.tensor(y_c, dtype=torch.float32)
x_d_tensor = torch.tensor(x_d, dtype=torch.float32, requires_grad=True)
y_d_tensor = torch.tensor(y_d, dtype=torch.float32, requires_grad=True)
t_d_tensor = torch.tensor(t_d, dtype=torch.float32)

# Visualizing boundary data points and collocation points
plt.title("Boundary data points and Collocation points")
plt.scatter(data[:, 0], data[:, 1], marker="x", c="k", label="BDP")
plt.scatter(colloc[:, 0], colloc[:, 1], s=2.2, marker=".", c="r", label="CP")
plt.xlabel("x",fontsize=16)
plt.ylabel("y",fontsize=16)
plt.axis("square")
plt.show()


##################################################################
######################## MODEL ###############################
##################################################################

# Definition of the Deep Neural Network (DNN) model
class DNN(nn.Module):
    def __init__(self, in_shape=(2, 1), out_shape=1, n_hidden_layers=7, neuron_per_layer=20, actfn=nn.Tanh):
        super(DNN, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.n_hidden_layers = n_hidden_layers
        self.neuron_per_layer = neuron_per_layer
        self.actfn = actfn()

        # Define the input layer
        self.input_layer = nn.Linear(in_shape[0] * in_shape[1], neuron_per_layer)

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(neuron_per_layer, neuron_per_layer))

        # Define the output layer
        self.output_layer = nn.Linear(neuron_per_layer, out_shape)

    def forward(self, x):
        # Reshape the input tensor
        x = x.view(-1, self.in_shape[0] * self.in_shape[1])

        # Pass through the input layer
        x = self.actfn(self.input_layer(x))

        # Pass through each hidden layer
        for layer in self.hidden_layers:
            x = self.actfn(layer(x))

        # Pass through the output layer
        x = self.output_layer(x)

        return x

# Instantiate the DNN model
model = DNN(in_shape=(2, 1), out_shape=1, n_hidden_layers=7, neuron_per_layer=20, actfn=nn.Tanh)


##################################################################
######################## LOSS FUNCTION ###########################
##################################################################

# Define the Physics-Informed Neural Network (PINN) loss function
def PINN_loss(model, x, t, f_exact):
    """
    Calculates the PINN loss given the model predictions, input coordinates, time values, and exact solution.
    
    Args:
        model (nn.Module): The neural network model
        x (torch.Tensor): Tensor containing the input coordinates
        t (torch.Tensor): Tensor containing the time values
        f_exact (torch.Tensor): Tensor containing the exact solution values
        
    Returns:
        torch.Tensor: Total loss value
    """
    # Make predictions using the model
    u_pred = model(torch.cat([x, t], dim=1))

    # Compute gradients
    u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u_pred.sum(), t, create_graph=True)[0]
    
    # Compute the residual
    residual = u_t - 0.1 * u_xx  
    
    # Compute the mean squared error (MSE) loss
    mse_loss = torch.mean((u_pred - f_exact)**2)
    
    # Compute the residual loss
    residual_loss = torch.mean(residual**2)
    
    # Total loss: weighted sum of MSE loss and residual loss
    total_loss = mse_loss + residual_loss
    
    return total_loss


##################################################################
########################## FUNCTIONS #############################
##################################################################

# Function to calculate the solution
def u(x, y):
    """
    Calculates the solution function u(x, y) using the neural network model.
    
    Args:
        x (torch.Tensor): Tensor containing x-coordinates
        y (torch.Tensor): Tensor containing y-coordinates
        
    Returns:
        torch.Tensor: Predicted solution values
    """
    inputs = torch.cat([x, y], dim=1) 
    return model(inputs)

# Function to calculate the x-derivative of the solution
def uderx(x, y):
    """
    Calculates the x-derivative of the solution function u(x, y) using the neural network model.
    
    Args:
        x (torch.Tensor): Tensor containing x-coordinates
        y (torch.Tensor): Tensor containing y-coordinates
        
    Returns:
        torch.Tensor: Predicted x-derivative values
    """
    inputs = torch.cat([x, y], dim=1)
    u = model(inputs)
    
    ones = torch.ones_like(u)
    uderx, = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)
    
    return uderx

# Function to calculate the y-derivative of the solution
def udery(x, y):
    """
    Calculates the y-derivative of the solution function u(x, y) using the neural network model.
    
    Args:
        x (torch.Tensor): Tensor containing x-coordinates
        y (torch.Tensor): Tensor containing y-coordinates
        
    Returns:
        torch.Tensor: Predicted y-derivative values
    """
    inputs = torch.cat([x, y], dim=1)
    u = model(inputs)
    
    ones = torch.ones_like(u)
    udery, = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)
    
    return udery

# Function to calculate the residual of the PDE
def f(x, y):
    """
    Calculates the residual of the partial differential equation (PDE) using the neural network model.
    
    Args:
        x (torch.Tensor): Tensor containing x-coordinates
        y (torch.Tensor): Tensor containing y-coordinates
        
    Returns:
        torch.Tensor: Residual values
    """
    x.requires_grad_(True)
    y.requires_grad_(True)

    u0 = u(x, y)
    u_x = torch.autograd.grad(u0.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u0.sum(), y, create_graph=True)[0]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    # Residual computation
    F = u_xx + u_yy + lam**2 * torch.exp(-u0)

    # Mean square of the residual
    return torch.mean(F**2)

# Function to ensure input is a tensor
def ensure_tensor(obj):
    """
    Ensures the input object is a torch.Tensor.
    
    Args:
        obj: Object to be converted to a tensor
        
    Returns:
        torch.Tensor: Converted tensor
    """
    if not isinstance(obj, torch.Tensor):
        return torch.tensor(obj, dtype=torch.float32)
    return obj

# Functions to compute mean squared errors for different parts of the solution
def msex(y, y_):
    """
    Computes the mean squared error (MSE) for the x-component of the solution.
    
    Args:
        y (torch.Tensor): Tensor containing the predicted solution values
        y_ (torch.Tensor): Tensor containing the true solution values
        
    Returns:
        torch.Tensor: MSE for the x-component of the solution
    """
    y = ensure_tensor(y)
    y_ = ensure_tensor(y_)
    yr = y.view(n_bc, n_data_per_bc)
    yr_ = y_.view(n_bc, n_data_per_bc)
    yrr = yr[0:2, :]
    yrr_ = yr_[0:2, :]
    retour = torch.mean((yrr - yrr_)**2)
    return retour

def msey(y, y_):
    """
    Computes the mean squared error (MSE) for the y-component of the solution.
    
    Args:
        y (torch.Tensor): Tensor containing the predicted solution values
        y_ (torch.Tensor): Tensor containing the true solution values
        
    Returns:
        torch.Tensor: MSE for the y-component of the solution
    """
    y = ensure_tensor(y)
    y_ = ensure_tensor(y_)
    yr = y.view(n_bc, n_data_per_bc)
    yr_ = y_.view(n_bc, n_data_per_bc)
    yrr = yr[2:4, :]
    yrr_ = yr_[2:4, :]
    retour = torch.mean((yrr - yrr_)**2)
    return retour

def msex1(y, y_):
    """
    Computes the mean squared error (MSE) for the x-component of the solution for boundary condition 1.
    
    Args:
        y (torch.Tensor): Tensor containing the predicted solution values
        y_ (torch.Tensor): Tensor containing the true solution values
        
    Returns:
        torch.Tensor: MSE for the x-component of the solution for boundary condition 1
    """
    y = ensure_tensor(y)
    y_ = ensure_tensor(y_)
    yr = y.view(n_bc, n_data_per_bc)
    yr_ = y_.view(n_bc, n_data_per_bc)
    yrr = yr[0:1, :]
    yrr_ = yr_[0:1, :]
    retour = torch.mean((yrr - yrr_)**2)
    return retour

def msex2(y, y_):
    """
    Computes the mean squared error (MSE) for the x-component of the solution for boundary condition 2.
    
    Args:
        y (torch.Tensor): Tensor containing the predicted solution values
        y_ (torch.Tensor): Tensor containing the true solution values
        
    Returns:
        torch.Tensor: MSE for the x-component of the solution for boundary condition 2
    """
    y = ensure_tensor(y)
    y_ = ensure_tensor(y_)
    yr = y.view(n_bc, n_data_per_bc)
    yr_ = y_.view(n_bc, n_data_per_bc)
    yrr = yr[1:2, :]  # Adjusted to target boundary condition 2
    yrr_ = yr_[1:2, :]
    return torch.mean((yrr - yrr_)**2)

def msey1(y, y_):
    """
    Computes the mean squared error (MSE) for the y-component of the solution for boundary condition 3.
    
    Args:
        y (torch.Tensor): Tensor containing the predicted solution values
        y_ (torch.Tensor): Tensor containing the true solution values
        
    Returns:
        torch.Tensor: MSE for the y-component of the solution for boundary condition 3
    """
    y = ensure_tensor(y)
    y_ = ensure_tensor(y_)
    yr = y.view(n_bc, n_data_per_bc)
    yr_ = y_.view(n_bc, n_data_per_bc)
    yrr = yr[2:3, :]  # Adjusted to target boundary condition 3
    yrr_ = yr_[2:3, :]
    return torch.mean((yrr - yrr_)**2)

def msey2(y, y_):
    """
    Computes the mean squared error (MSE) for the y-component of the solution for boundary condition 4.
    
    Args:
        y (torch.Tensor): Tensor containing the predicted solution values
        y_ (torch.Tensor): Tensor containing the true solution values
        
    Returns:
        torch.Tensor: MSE for the y-component of the solution for boundary condition 4
    """
    y = ensure_tensor(y)
    y_ = ensure_tensor(y_)
    yr = y.view(n_bc, n_data_per_bc)
    yr_ = y_.view(n_bc, n_data_per_bc)
    yrr = yr[3:4, :]  # Adjusted to target boundary condition 4
    yrr_ = yr_[3:4, :]
    return torch.mean((yrr - yrr_)**2)

# Initialization of variables and optimization loop
loss = 0
epochs = 60000
loss_values = np.array([])
L_values = np.array([])
l_values = np.array([])
lp_values = np.array([])

start = time.time()
        
optimizer = optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(epochs):
    # Compute predictions and derivatives for boundary conditions
    T_ = u(x_d_tensor, y_d_tensor)
    Tx_ = uderx(x_d_tensor, y_d_tensor)
    Ty_ = udery(x_d_tensor, y_d_tensor)

    # Compute PDE loss and boundary condition loss
    L = f(x_c_tensor, y_c_tensor)
    l = 0 * msex1(t_d, T_) + 0 * msex2(t_d, T_) + 1 * msey1(t_d, T_) + 0 * msey2(t_d, T_)
    l = l + 0 * msex1(t_dy, Ty_) + 0 * msex2(t_dy, Ty_) + 1 * msey1(t_dy, Ty_) + 0 * msey2(t_dy, Ty_)
    
    # Total loss
    loss = L + l

    # Reset gradients to zero for each optimization step
    optimizer.zero_grad()

    # Compute gradients of 'loss' with respect to model parameters
    loss.backward()

    # Apply computed gradients to the model's parameters
    optimizer.step()

    # Append current loss values to arrays for tracking
    loss_values = np.append(loss_values, loss.item()) 
    lp_values = np.append(lp_values, loss.item())
    
    # Print progress every 100 epochs
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"{epoch:5}, {loss.item():.8f}")
        L_values = np.append(L_values, L.item())
        l_values = np.append(l_values, l.item())

# Calculation of computation time
end = time.time()
computation_time = end - start
print(f"\ncomputation time: {computation_time:.3f}\n")


##################################################################
########################### VISUALIZATION ########################
##################################################################

# Defining save path for saving graphs
save_path = r"C:\Users\kerim\Desktop\Graphs" 

# Setup for the heat maps
n = 100
l = 1.
r = 2 * l / (n + 1)
T = np.zeros([n * n, n * n])

# First plot - Loss values over epochs
plt.figure(figsize=(10, 5))
plt.semilogy(loss_values, label="model")
plt.semilogy(lp_values, label="Total loss")
plt.xlabel("Epochs ($\\times 10^2$)", fontsize=16)
plt.legend()
plt.savefig(f"{save_path}/Epochs.png")  # Save plot as an image
plt.show()

# Heat map and error plotting
plt.figure(figsize=(14, 14))

X = np.linspace(-lim1, lim1, n)
Y = np.linspace(0., lim2, n)
X0, Y0 = np.meshgrid(X, Y)
X = X0.reshape([n * n, 1])
Y = Y0.reshape([n * n, 1])
X_T = torch.Tensor(X)
Y_T = torch.Tensor(Y)

S = u(X_T, Y_T).detach().numpy().reshape(n, n)
TT = tru(X0, Y0)

# Plotting PINN solution
plt.subplot(221)
plt.pcolormesh(X0, Y0, S, cmap="turbo")
plt.colorbar(pad=0.1)
plt.xlabel("x", fontsize=16)
plt.ylabel("z", fontsize=16)
plt.title("PINN solution")
plt.axis("square")

# Plotting PINN error
plt.subplot(222)
plt.pcolormesh(X0, Y0, S - TT, cmap="turbo")
plt.colorbar(pad=0.1)
plt.title("PINN error")
plt.xlabel("x", fontsize=16)
plt.ylabel("z", fontsize=16)
plt.axis("square")

# Plotting Exact solution
plt.subplot(223)
plt.pcolormesh(X0, Y0, TT, cmap="turbo")
plt.colorbar(pad=0.1)
plt.xlabel("x", fontsize=16)
plt.ylabel("z", fontsize=16)
plt.title("Exact solution")
plt.axis("square")

# Plotting Error
plt.subplot(224)
plt.pcolormesh(X0, Y0, TT - S, cmap="turbo")
plt.colorbar(pad=0.1)
plt.title("Error")
plt.xlabel("x", fontsize=16)
plt.ylabel("z", fontsize=16)
plt.axis("square")
plt.tight_layout()
plt.savefig(f"{save_path}/HeatMap.png")  # Save plot as an image
plt.show()

# Plotting PINN solution with contour lines
plt.figure(figsize=(14, 7))
plt.contour(X0, Y0, S, 24, linewidths=2)
plt.colorbar(pad=0.1)
plt.scatter(data[:, 0], data[:, 1], marker=".", c="r", s=200, label="BDP")
plt.scatter(colloc[:, 0], colloc[:, 1], marker=".", c="b", s=25, label="CP")
plt.title("PINNs solution", fontsize=16)
plt.xlabel("x", fontsize=16)
plt.ylabel("z", fontsize=16)
plt.xlim(-lim1, lim1)
plt.ylim(0., lim2)
plt.axis("square")
plt.tight_layout()
plt.savefig(f"{save_path}/PINNsSoln.png")  # Save plot as an image
plt.show()

# Plotting Exact solution with contour lines
plt.figure(figsize=(14, 7))
plt.contour(X0, Y0, S, 24, linewidths=2)
plt.colorbar(pad=-0.25)
plt.title(r"Exact solution", fontsize=16)
plt.xlabel("x",fontsize=16)
plt.ylabel("z",fontsize=16)
plt.xlim(-lim1, lim1)
plt.ylim(0., lim2)
plt.tight_layout()
plt.axis("square")
plt.savefig(f"{save_path}/ExactSoln.png")  # Save plot as an image
plt.show()

# Loss plots
plt.figure(figsize=(10, 5))
plt.semilogy(l_values, label='Loss_data')
plt.semilogy(L_values, label='Loss_PDE')
plt.xlabel("Epochs ($\\times 10^2$)", fontsize=16)
plt.legend()
plt.savefig(f"{save_path}/Loss.png")  # Save plot as an image
plt.show()
