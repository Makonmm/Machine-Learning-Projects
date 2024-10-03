import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

# Dense layer class


class Dense:
    def __init__(self, feat_size, out_size):
        """Initialize Dense layer with weights and bias."""
        self.feat_size = feat_size
        self.out_size = out_size
        self.weights = np.random.normal(
            0, 1, (feat_size, out_size)) * np.sqrt(2 / feat_size)
        self.bias = np.random.rand(1, out_size) - 0.5

    def forward(self, input_data):
        """Forward pass through the layer."""
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_der, lr):
        """Backward pass for gradient descent."""
        input_der = np.dot(output_der, self.weights.T)
        weight_der = np.dot(self.input.T.reshape(-1, 1), output_der)
        self.weights -= lr * weight_der
        self.bias -= lr * output_der
        return input_der

# Activation functions


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)

# Activation layer class


class ActLayer:
    def __init__(self, act, act_prime):
        """Initialize Activation layer."""
        self.act = act
        self.act_prime = act_prime

    def forward(self, input_data):
        """Forward pass through the activation layer."""
        self.input = input_data
        self.output = self.act(self.input)
        return self.output

    def backward(self, output_der, lr):
        """Backward pass for activation layer."""
        return self.act_prime(self.input) * output_der

# Mean Squared Error function


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Neural Network class


class Network:
    def __init__(self, loss, loss_prime):
        """Initialize the Neural Network."""
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def predict(self, input_data):
        """Predict output for input data."""
        return [self._forward(x) for x in input_data]

    def _forward(self, x):
        """Forward pass for a single input."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, X_train, y_train, epochs, lr):
        """Train the neural network."""
        for epoch in range(epochs):
            total_error = 0
            for j in range(len(X_train)):
                layer_output = X_train[j]
                for layer in self.layers:
                    layer_output = layer.forward(layer_output)
                total_error += self.loss(y_train[j], layer_output)
                gradient = self.loss_prime(y_train[j], layer_output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, lr)
            avg_error = total_error / len(X_train)
            print(f"Epoch {epoch + 1}/{epochs} Error = {avg_error:.6f}")


def simulate_data(s0, r, steps, maturity, vol):
    """Simulate asset price data."""
    delta_t = maturity / steps
    prices = [s0]
    normal_dist = np.random.normal(0, np.sqrt(delta_t), steps)

    for x in range(steps):
        prices.append(calculate_spot(
            prices[-1], vol, r, delta_t, normal_dist[x]))
    return prices


def calculate_spot(prev, sigma, r, step, random):
    """Calculate the spot price."""
    return (prev + (sigma * prev * random) + (r * prev * step))


def main():
    # Data preparation
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]).reshape(-1, 2)
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]]).reshape(-1, 1)

    # Initialize and train the model
    xor_model = Network(mse, mse_prime)
    xor_model.add(Dense(2, 3))
    xor_model.add(ActLayer(relu, relu_prime))
    xor_model.add(Dense(3, 1))
    xor_model.fit(x_train, y_train, epochs=2000, lr=0.01)

    # Predictions
    y_pred = xor_model.predict(x_train)
    print("Real values:", list(y_train.reshape(-1)))
    print("Predicted values:", [round(float(x)) for x in y_pred])

    # Variables for simulations
    vol = 0.17
    T = 1/2
    n = 1000
    s_0 = 100
    r = 0.05
    k = 100

    # Simulate asset prices
    sims = pd.DataFrame(
        {f'Sim_{i+1}': simulate_data(s_0, r, n, T, vol) for i in range(5)})
    sims.index = np.round(np.arange(0, 0.5 + (0.5 / n), 0.5 / n), 4)

    # Plot simulations
    sns.set_theme(style='whitegrid', font_scale=2.5)
    plt.figure(figsize=(40, 18))
    ax = sns.lineplot(data=sims, palette='bright', linewidth=2.5)
    ax.set(xlabel='Stages', ylabel='Prices', title='Simulations')
    plt.savefig(os.path.join('FinancialAssetDL/graphs',
                'simulations_graph.png'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
