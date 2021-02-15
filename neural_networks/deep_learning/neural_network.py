import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Activation function.

    Sigmoid f(x) = 1 / (1 + e^(-x))
    """
    x = 1 / (1 + np.exp(-x))
    return x


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    d(sigmoid(x)) / d(x) = x * (1 - x)
    """
    return x * (1 - x)


class SimpleNeuralNetwork:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        hidden_layer_size: int = 4,
        alpha: float = 1.0,
    ):
        self.alpha: float = alpha
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y
        self.weights_1: np.ndarray = np.random.rand(self.X.shape[1], hidden_layer_size)
        self.hidden_layer: np.ndarray = None
        self.weights_2: np.ndarray = np.random.rand(hidden_layer_size, 1)
        self.Y_predicted: np.ndarray = np.zeros(self.Y.shape)

    def feed(self):
        """
        Compute the output of the hidden layers and the output layer
        """
        self.hidden_layer = sigmoid(np.dot(self.X, self.weights_1))
        self.Y_predicted = sigmoid(np.dot(self.hidden_layer, self.weights_2))

    def predict(self):
        pass

    def update_weights(self):
        """
        Update the weights on every iteration using the Backpropagation algorithm
        """
        delta_weights_2 = np.dot(
            self.hidden_layer.T,
            (self.Y_predicted - self.Y) * sigmoid_derivative(self.Y_predicted),
        )
        delta_weights_1 = np.dot(
            self.X.T,
            np.dot(
                (self.Y_predicted - self.Y) * sigmoid_derivative(self.Y_predicted),
                self.weights_2.T,
            )
            * sigmoid_derivative(self.hidden_layer),
        )

        self.weights_1 = self.weights_1 - self.alpha * delta_weights_1
        self.weights_2 = self.weights_2 - self.alpha * delta_weights_2

    @property
    def loss(self):
        """
        Mean Squared Error
        """
        return np.mean(np.square(self.Y - self.Y_predicted))

    def train(self):
        self.feed()
        self.update_weights()


if __name__ == "__main__":
    X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
    y = np.array(([0], [1], [1], [1]), dtype=float)

    neural_network = SimpleNeuralNetwork(X, y, hidden_layer_size=4, alpha=1)
    epoch = 1000

    for i in range(epoch):
        if i % 100 == 0:
            print(f"-- Iteration: {i} --")
            print(f"X: {neural_network.X}")
            print(f"Y: {neural_network.Y}")
            print(f"Y_predicted: {neural_network.Y_predicted}")
            print(f"Loss: {neural_network.loss}")
            print("\n")

        neural_network.train()
