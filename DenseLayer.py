import numpy as np

class Dense_SingleOutput:
    def __init__(self, input_dim, output_dim=1, verbose=False):

        self.input_dim = input_dim
        self.output_dim = 1

        # 0.1 ~ 1.1 | 0.1 is to avoid 0 weights and biases
        self.bias = np.round(np.random.random() + 0.1, 2)
        self.weights = np.round(np.random.random(input_dim) + 0.1, 2)

        if verbose:
            print(
                f"Dense_SingleOutput layer with input_dim: {self.input_dim} and output_dim: 1"
            )
            print(f"Initial weights: {self.weights}")
            print(f"Initial bias: {self.bias}")

    def forward(self, x):
        return np.dot(self.weights, x.T) + self.bias


class Optimizer:
    def __init__(
        self,
        param,
        lr=0.01,
        beta_1=0.9,
        beta_2=0.999,
    ):
        # ADAM
        self.param = param
        self.beta_m = beta_1
        self.beta_v = beta_2
        self.lr =lr
        
        # other parameters
        self.momentum = 0
        self.prev_momentum = 0
        self.v = 0
        self.prev_v = 0

    def step(self, gradient, epoch):
        self.prev_momentum = self.momentum
        self.prev_v = self.v

        # calculate the moving average of the momentum and the squared gradients
        self.momentum = (self.beta_m * self.prev_momentum) + (
            1 - self.beta_m
        ) * gradient
        self.v = (self.beta_v * self.prev_v) + (1 - self.beta_v) * (gradient**2)

        # bias correction
        momentum_hat = self.momentum / (1 - self.beta_m ** (epoch + 1))
        v_hat = self.v / (1 - self.beta_v ** (epoch + 1))

        learning_rate = self.lr / (np.sqrt(v_hat) + 1e-8)
        self.param -= learning_rate * momentum_hat

    def get_param(self):
        return self.param


class MSELoss:
    def __init__(self):
        pass

    def calculate_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_gradient_coeff(self, y_true, y_pred, X):
        errors = y_true - y_pred
        return (2 / len(y_true)) * np.dot(errors, X)

    def calculate_gradient_bias(self, y_true, y_pred):
        errors = y_true - y_pred
        return (2 / len(y_true)) * np.sum(errors)


class Model:
    def __init__(self, layer, loss, optimizer):
        self.layer = layer  # not initialized
        self.loss = loss
        self.optimizer = optimizer
        self.initiated = False

    def _forward(self, x):
        if not self.initiated:
            raise Exception("Model not initiated - call `fit()` method first")
        
        return self.layer.forward(x)

    def _backward(self, y_pred, y_real, X):
        if not self.initiated:
            raise Exception("Model not initiated - call `fit()` method first")
        
        loss = self.loss.calculate_loss(y_pred, y_real)
        gradient_coeff = self.loss.calculate_gradient_coeff(y_pred, y_real, X)
        gradient_bias = self.loss.calculate_gradient_bias(y_pred, y_real)

        return loss, gradient_coeff, gradient_bias

    def fit(self, X, y, epochs=100):
        input_dim = X.shape[-1]
        self.layer = self.layer(input_dim=input_dim, verbose=True)
        self.w_optimizers = [self.optimizer(w) for w in self.layer.weights]
        self.b_optimizer = self.optimizer(self.layer.bias)
        self.initiated = True
        ### ------------------- ###

        for epoch in range(epochs):
            for i, x_i in enumerate(X):

                # ---------Forward pass---------
 
                y_pred = self._forward(X)

                # ---------Backward pass---------
                
                # Calculate loss and gradients
                loss, gradient_coeff, gradient_bias = self._backward(y_pred, y, X)

                # # Update weights and biases
                for w_idx, w_optimzier in enumerate(self.w_optimizers):
                    w_optimzier.step(gradient_coeff[w_idx], epoch)
                self.b_optimizer.step(gradient_bias, epoch)
                
                # Fetch the updated weights and biases
                self.layer.weights = np.array(
                    [w.get_param() for w in self.w_optimizers]
                )
                self.layer.bias = self.b_optimizer.get_param()

            print(f"Epoch: {epoch} | Loss: {loss:.4f}")


model = Model(layer=Dense_SingleOutput, loss=MSELoss(), optimizer=Optimizer)
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([4, 8, 12])
model.fit(X_train, y_train, epochs=100)


