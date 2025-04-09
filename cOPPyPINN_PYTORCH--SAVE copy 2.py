import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Fix random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Define the neural network for the PINN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.fc = nn.Sequential()
        for i in range(len(layers) - 1):
            self.fc.add_module(f'layer_{i}', nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.fc.add_module(f'activation_{i}', nn.Tanh())
    
    def forward(self, x):
        return self.fc(x)

# Function to compute the residuals
def compute_residuals(model, x, y):
    uvp = model(torch.cat([x, y], dim=1))
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
    
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
    
    momentum_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    continuity = u_x + v_y
    
    return momentum_u, momentum_v, continuity, u, v, p

# Loss function
def compute_loss(model, x, y, boundary_x, boundary_y, boundary_u, boundary_v):
    momentum_u, momentum_v, continuity, u, v, p = compute_residuals(model, x, y)
    loss_pde = torch.mean(momentum_u**2) + \
               torch.mean(momentum_v**2) + \
               torch.mean(continuity**2)
    
    uvp_pred = model(torch.cat([boundary_x, boundary_y], dim=1))
    u_pred, v_pred, _ = torch.split(uvp_pred, 1, dim=1)
    loss_bc = torch.mean((u_pred - boundary_u)**2) + \
              torch.mean((v_pred - boundary_v)**2)
    
    return loss_pde + loss_bc

# Training function
def train(model, optimizer, x, y, boundary_x, boundary_y, boundary_u, boundary_v, epochs, print_every=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, x, y, boundary_x, boundary_y, boundary_u, boundary_v)
        loss.backward()
        optimizer.step()
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model state and optimizer state after training
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "pinn_model_checkpoint.pth")
    print("Model saved successfully!")

# Generate collocation points (domain and boundary)
def generate_points(n_domain, n_boundary):
    x_domain = np.random.uniform(0, 1, (n_domain, 1))
    y_domain = np.random.uniform(0, 1, (n_domain, 1))
    
    x_boundary = np.concatenate([np.zeros((n_boundary, 1)),
                                  np.ones((n_boundary, 1)),
                                  np.linspace(0, 1, n_boundary)[:, None],
                                  np.linspace(0, 1, n_boundary)[:, None]])
    y_boundary = np.concatenate([np.linspace(0, 1, n_boundary)[:, None],
                                  np.linspace(0, 1, n_boundary)[:, None],
                                  np.ones((n_boundary, 1)),
                                  np.zeros((n_boundary, 1))])
    u_boundary = np.concatenate([np.zeros((n_boundary, 1)),
                                  np.zeros((n_boundary, 1)),
                                  np.ones((n_boundary, 1)),
                                  np.zeros((n_boundary, 1))])
    v_boundary = np.zeros_like(u_boundary)
    
    return x_domain, y_domain, x_boundary, y_boundary, u_boundary, v_boundary

# Streamplot visualization
def plot_results(model, n_plot=100):
    model.eval()  # Set the model to evaluation mode
    x = np.linspace(0, 1, n_plot)
    y = np.linspace(0, 1, n_plot)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the meshgrid and convert to tensors
    x_flat = X.flatten()[:, None]
    y_flat = Y.flatten()[:, None]
    x_tensor = torch.tensor(x_flat, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32, requires_grad=True)
    
    # Forward pass to get u, v, p
    with torch.no_grad():
        uvp = model(torch.cat([x_tensor, y_tensor], dim=1))
        u = uvp[:, 0].detach().numpy().reshape(X.shape)  # Detach and convert to NumPy
        v = uvp[:, 1].detach().numpy().reshape(X.shape)  # Detach and convert to NumPy
        p = uvp[:, 2].detach().numpy().reshape(X.shape)  # Detach and convert to NumPy
    
    # Compute vorticity (requires gradients)
    x_tensor.requires_grad_(True)
    y_tensor.requires_grad_(True)
    uvp = model(torch.cat([x_tensor, y_tensor], dim=1))
    u_grad = uvp[:, 0:1]
    v_grad = uvp[:, 1:2]
    
    # Compute gradients for vorticity
    du_dy = torch.autograd.grad(u_grad, y_tensor, torch.ones_like(u_grad), retain_graph=True)[0]
    dv_dx = torch.autograd.grad(v_grad, x_tensor, torch.ones_like(v_grad))[0]
    vorticity = (dv_dx - du_dy).detach().numpy().reshape(X.shape)  # Detach and convert to NumPy
    
    # Plotting
    plt.figure(figsize=(20, 15))
    
    # Streamplot with velocity magnitude
    plt.subplot(2, 2, 1)
    speed = np.sqrt(u**2 + v**2)
    strm = plt.streamplot(X, Y, u, v, color=speed, cmap='jet', density=2)
    plt.colorbar(strm.lines, label='Velocity Magnitude')
    plt.title('Streamlines and Velocity Magnitude')
    
    # Pressure field
    plt.subplot(2, 2, 2)
    plt.contourf(X, Y, p, levels=50, cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Distribution')
    
    # Vorticity
    plt.subplot(2, 2, 3)
    plt.contourf(X, Y, vorticity, levels=50, cmap='coolwarm')
    plt.colorbar(label='Vorticity')
    plt.title('Vorticity Field')
    
    # Velocity components
    plt.subplot(2, 2, 4)
    plt.quiver(X[::5, ::5], Y[::5, ::5], u[::5, ::5], v[::5, ::5])
    plt.title('Velocity Vector Field')
    
    plt.tight_layout()
    plt.show()

# Function to extract velocity profiles along vertical and horizontal lines
def extract_velocity_profiles(model, x, y):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        uvp = model(torch.cat([x, y], dim=1))
        u = uvp[:, 0].detach().numpy()  # Detach and convert to NumPy
        v = uvp[:, 1].detach().numpy()  # Detach and convert to NumPy
    return u, v

# Function to plot velocity profiles
def plot_velocity_profiles(model, n_plot=100):
    model.eval()  # Set the model to evaluation mode
    
    # Vertical line through geometric center (x = 0.5)
    x_vertical = np.full((n_plot, 1), 0.5)
    y_vertical = np.linspace(0, 1, n_plot)[:, None]
    x_vertical_tensor = torch.tensor(x_vertical, dtype=torch.float32)
    y_vertical_tensor = torch.tensor(y_vertical, dtype=torch.float32)
    u_vertical, v_vertical = extract_velocity_profiles(model, x_vertical_tensor, y_vertical_tensor)
    
    # Horizontal line through geometric center (y = 0.5)
    x_horizontal = np.linspace(0, 1, n_plot)[:, None]
    y_horizontal = np.full((n_plot, 1), 0.5)
    x_horizontal_tensor = torch.tensor(x_horizontal, dtype=torch.float32)
    y_horizontal_tensor = torch.tensor(y_horizontal, dtype=torch.float32)
    u_horizontal, v_horizontal = extract_velocity_profiles(model, x_horizontal_tensor, y_horizontal_tensor)
    
    # Plotting
    plt.figure(figsize=(20, 10))
    
    # U-velocity along vertical line through geometric center
    plt.subplot(2, 2, 1)
    plt.plot(y_vertical, u_vertical, label='U-velocity')
    plt.xlabel('Y')
    plt.ylabel('U-velocity')
    plt.title('U-velocity along vertical line through geometric center (x = 0.5)')
    plt.legend()
    
    # V-velocity along horizontal line through geometric center
    plt.subplot(2, 2, 2)
    plt.plot(x_horizontal, v_horizontal, label='V-velocity')
    plt.xlabel('X')
    plt.ylabel('V-velocity')
    plt.title('V-velocity along horizontal line through geometric center (y = 0.5)')
    plt.legend()
    
    # U-velocity along vertical line through primary vortex center (x = 0.5)
    x_vertical_vortex = np.full((n_plot, 1), 0.5)
    y_vertical_vortex = np.linspace(0, 1, n_plot)[:, None]
    x_vertical_vortex_tensor = torch.tensor(x_vertical_vortex, dtype=torch.float32)
    y_vertical_vortex_tensor = torch.tensor(y_vertical_vortex, dtype=torch.float32)
    u_vertical_vortex, v_vertical_vortex = extract_velocity_profiles(model, x_vertical_vortex_tensor, y_vertical_vortex_tensor)
    
    plt.subplot(2, 2, 3)
    plt.plot(y_vertical_vortex, u_vertical_vortex, label='U-velocity')
    plt.xlabel('Y')
    plt.ylabel('U-velocity')
    plt.title('U-velocity along vertical line through primary vortex center (x = 0.5)')
    plt.legend()
    
    # V-velocity along horizontal line through primary vortex center (y = 0.5)
    x_horizontal_vortex = np.linspace(0, 1, n_plot)[:, None]
    y_horizontal_vortex = np.full((n_plot, 1), 0.5)
    x_horizontal_vortex_tensor = torch.tensor(x_horizontal_vortex, dtype=torch.float32)
    y_horizontal_vortex_tensor = torch.tensor(y_horizontal_vortex, dtype=torch.float32)
    u_horizontal_vortex, v_horizontal_vortex = extract_velocity_profiles(model, x_horizontal_vortex_tensor, y_horizontal_vortex_tensor)
    
    plt.subplot(2, 2, 4)
    plt.plot(x_horizontal_vortex, v_horizontal_vortex, label='V-velocity')
    plt.xlabel('X')
    plt.ylabel('V-velocity')
    plt.title('V-velocity along horizontal line through primary vortex center (y = 0.5)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Hyperparameters
nu = 0.01
n_domain = 10000
n_boundary = 200
layers = [2, 20, 20, 20, 3]
lr = 0.001
epochs = 1000

# Generate training points
x_domain, y_domain, x_boundary, y_boundary, u_boundary, v_boundary = generate_points(n_domain, n_boundary)

x_domain = torch.tensor(x_domain, dtype=torch.float32, requires_grad=True)
y_domain = torch.tensor(y_domain, dtype=torch.float32, requires_grad=True)
x_boundary = torch.tensor(x_boundary, dtype=torch.float32, requires_grad=True)
y_boundary = torch.tensor(y_boundary, dtype=torch.float32, requires_grad=True)
u_boundary = torch.tensor(u_boundary, dtype=torch.float32)
v_boundary = torch.tensor(v_boundary, dtype=torch.float32)

# Initialize model and optimizer
model = PINN(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Check for a checkpoint and load model and optimizer states if available
checkpoint_path = "pinn_model_checkpoint.pth"
if os.path.exists(checkpoint_path):
    print("Checkpoint found! Loading model and optimizer states...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model and optimizer states loaded successfully!")
else:
    print("No checkpoint found. Training will start from scratch.")

# Train the model
train(model, optimizer, x_domain, y_domain, x_boundary, y_boundary, u_boundary, v_boundary, epochs, print_every=500)

# Plot the results
plot_results(model)
plot_velocity_profiles(model)