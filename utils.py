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
        return 2 / len(y) * np.dot(errors, x)

    def get_bias_gradient(self, y_pred, y):
        y_pred = self.make_np(y_pred)
        y = self.make_np(y)

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred.shape} and y shape {y.shape} must be the same."
            )

        errors = y_pred - y
        return 2 / len(y) * np.sum(errors)
    


class Adam:
    def __init__(self, param, beta_m=0.9, beta_v=0.999):
        self.param = param
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.momentum = 0
        self.prev_momentum = 0
        self.lr = 0.09
        self.v = 0
        self.prev_v = 0

    def step(self, gradient, epoch):
        # Store previous momentum and velocity
        self.prev_momentum = self.momentum
        self.prev_v = self.v

        # Calculate the moving average of the momentum and the squared gradients
        self.momentum = (self.beta_m * self.prev_momentum) + (1 - self.beta_m) * gradient
        self.v = (self.beta_v * self.prev_v) + (1 - self.beta_v) * (gradient**2)

        # Bias correction
        momentum_hat = self.momentum / (1 - self.beta_m ** (epoch + 1))
        v_hat = self.v / (1 - self.beta_v ** (epoch + 1))

        # Update parameter with corrected values
        learning_rate = self.lr / (np.sqrt(v_hat) + 1e-8)
        self.param -= learning_rate * momentum_hat

    def get_param(self):
        return self.param