# %%
import numpy as np

# %%
class CostFunction:
    def __init__(self):
        pass

    def get_loss(self, y_pred, y):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_weight_gradient(self, y_pred, y, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_bias_gradient(self, y_pred, y):
        raise NotImplementedError("Subclasses should implement this method.")

    def make_np(self, arr):
        return np.asarray(arr)


class MSE(CostFunction):
    def __init__(self):
        super().__init__()

    def get_loss(self, y_pred, y):
        y_pred = self.make_np(y_pred)
        y = self.make_np(y)

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred.shape} and y shape {y.shape} must be the same."
            )

        errors = y_pred - y
        return np.mean(errors**2)

    def get_weight_gradient(self, y_pred, y, x):
        y_pred = self.make_np(y_pred)
        y = self.make_np(y)

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred.shape} and y shape {y.shape} must be the same."
            )

        errors = y_pred - y
        return 2 / len(y) * np.sum(errors * x)

    def get_bias_gradient(self, y_pred, y):
        y_pred = self.make_np(y_pred)
        y = self.make_np(y)

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred.shape} and y shape {y.shape} must be the same."
            )

        errors = y_pred - y
        return 2 / len(y) * np.sum(errors)

# %%
class Adam:
    def __init__(self, lr):
        pass

# %%
class Dense:
    def __init__(self, units: int):
        self.units: int = units
        self.w = None
        self.b = None
        self.initialized = False

    def initialize_params(self, output_dim):
        self.w = np.random.rand(output_dim, self.units)
        self.b = np.random.rand(output_dim)
        self.initialized = True

    def forward(self, X):
        if not self.initialized:
            raise ValueError("Initiate params first")        
        X = np.asarray(X)
        return np.dot(X , self.w.T) + self.b


class Input(Dense):
    def __init__(self, units):
        super().__init__(units)
        self.units = units


class Output():
    def __init__(self, units):
        self.units = units

# %%

class Sequential:
    def __init__(self, layers: list, verbose=False):
        self.layers = layers
        self.hidden_layers = self.layers[1:-1]
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        self.verbose = verbose
        self.cost_function = None
        self.optimizer = None

        self.validate_input()
        self.validate_output()

        self.is_compiled = False

    def fit(self, X, y, epochs=1):
        if not self.is_compiled:
            raise ValueError("Call `compile()` method first.")

        self.initialize_layers()

        loss = None
        for epoch in range(epochs):
            print(f"Epoch: {epoch}| Loss: ", end="\r")
            
            y_pred = self.forward(X)
            loss = self.cost_function.get_loss(y, y_pred)
            print(loss)
        
    def forward(self, X):
        i = 0
        y_pred = X
        while i < len(self.layers):
            layer = self.layers[i]
            if isinstance(layer, Output):
                break
            y_pred = layer.forward(y_pred)
            
            i += 1            
        return y_pred        
            # X = self.layers[0].forward(X)
            # print("input -> dense 1", X.shape)
            # X = self.layers[1].forward(X)
            # print("dense 1 -> dense 2", X.shape)
            # X = self.layers[2].forward(X)
            # print("dense 2 ->  output", X.shape)

    def initialize_layers(self):
        i = 0
        while i < len(self.layers):

            layer = self.layers[i]
            if isinstance(layer, Output):
                break

            output_dim = self.layers[i + 1].units
            layer.initialize_params(output_dim)

            i += 1
            
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def compile(self, cost_function, optimizer):

        self.cost_function = cost_function
        self.optimizer = optimizer
        if self.verbose:
            print(self)
            print(f"Loss: {self.cost_function.__class__.__name__}")
            print(f"Optimizer: {self.optimizer.__class__.__name__}")

        self.is_compiled = True

    def __str__(self):
        output = "|  Sequential Model   |\n\n"
        for idx, layer in enumerate(self.layers):
            if not isinstance(layer, (Input, Output)):
                output += " -> "
            output += f"[{idx}] {layer.__class__.__name__} ({layer.units})\n"

        return output

    def validate_input(self):
        if not isinstance(self.input_layer, Input):
            raise TypeError(
                f"The first layer must be an object of class `Input`, got object of class `{self.input_layer.__class__.__name__}`"
            )

    def validate_output(self):
        if not isinstance(self.output_layer, Output):
            raise TypeError(
                f"The last layer must be an object of class `Output`, got object of class `{self.output_layer.__class__.__name__}`"
            )

# %%
n_samples = 500
X_data = np.random.normal(0, 1, size=(n_samples, 2))
y_data = np.array([[x1 * 0.3 + 4, x2 * 1.9 - 8] for x1, x2 in X_data])

# %%
model = Sequential([Input(2), Dense(3), Dense(5), Output(2)])
model.compile(cost_function=MSE(), optimizer=Adam(0.01))
model.fit(X_data, y_data)


