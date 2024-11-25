import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        limit1 = np.sqrt(6 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        limit2 = np.sqrt(6 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("why?")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError("why?")

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = self.output_activation(self.z2)
        return self.out

    def backward(self, X, y):
        N = X.shape[0]
        epsilon = 1e-15
        dLoss_out = -(y / (self.out + epsilon)) + ((1 - y) / (1 - self.out + epsilon))
        dLoss_out *= self.out * (1 - self.out)
        dLoss_out /= N

        self.dW2 = self.a1.T @ dLoss_out
        self.db2 = np.sum(dLoss_out, axis=0, keepdims=True)

        dLoss_a1 = dLoss_out @ self.W2.T
        dLoss_z1 = dLoss_a1 * self.activation_derivative(self.z1)

        self.dW1 = X.T @ dLoss_z1
        self.db1 = np.sum(dLoss_z1, axis=0, keepdims=True)

        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

    def output_activation(self, x):
        return 1 / (1 + np.exp(-x))

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1).astype(int).reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)

    W2 = mlp.W2.ravel()
    b2 = mlp.b2.ravel()
    xx, yy = np.meshgrid(np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 10),
                         np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 10))
    if W2[2] != 0:
        zz = (-W2[0] * xx - W2[1] * yy - b2[0]) / W2[2]
        ax_hidden.plot_surface(xx, yy, zz, alpha=0.3)
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    grid_xx, grid_yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                                   np.linspace(y_min, y_max, 30))
    grid_points = np.c_[grid_xx.ravel(), grid_yy.ravel()]
    z1_grid = grid_points @ mlp.W1 + mlp.b1
    a1_grid = mlp.activation(z1_grid)
    ax_hidden.scatter(a1_grid[:, 0], a1_grid[:, 1], a1_grid[:, 2],
                      c='gray', alpha=0.1, marker='.')

    z2_grid = a1_grid @ mlp.W2 + mlp.b2
    z2_grid = z2_grid.reshape(grid_xx.shape)
    ax_input.contourf(grid_xx, grid_yy, z2_grid, levels=[-np.inf, 0, np.inf],
                      colors=['blue', 'red'], alpha=0.3)
    ax_input.contour(grid_xx, grid_yy, z2_grid, levels=[0], colors='k')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_xlabel('Input Feature 1')
    ax_input.set_ylabel('Input Feature 2')

    neuron_positions = {
        'input': [(0, i) for i in range(2)],
        'hidden': [(1, i) for i in range(3)],
        'output': [(2, 0)]
    }

    for layer, positions in neuron_positions.items():
        for pos in positions:
            circle = Circle(pos, radius=0.1, fill=True, color='lightgray')
            ax_gradient.add_patch(circle)
            ax_gradient.text(pos[0], pos[1], layer[0].upper(), ha='center', va='center')

    for i, (x0, y0) in enumerate(neuron_positions['input']):
        for j, (x1, y1) in enumerate(neuron_positions['hidden']):
            gradient = mlp.dW1[i, j]
            linewidth = np.abs(gradient) * 10
            ax_gradient.plot([x0, x1], [y0, y1], 'k-', linewidth=linewidth, alpha=0.5)

    for i, (x0, y0) in enumerate(neuron_positions['hidden']):
        x1, y1 = neuron_positions['output'][0]
        gradient = mlp.dW2[i, 0]
        linewidth = np.abs(gradient) * 10
        ax_gradient.plot([x0, x1], [y0, y1], 'k-', linewidth=linewidth, alpha=0.5)

    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-0.5, len(neuron_positions['hidden']) - 0.5)
    ax_gradient.axis('off')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                     ax_gradient=ax_gradient, X=X, y=y),
                        frames=step_num // 10, repeat=False)

    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
