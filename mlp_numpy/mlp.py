from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}


@dataclass
class ForwardCache:
    x: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    a2: np.ndarray
    z3: np.ndarray
    probs: np.ndarray


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] | list[int],
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        if len(hidden_dims) != 2:
            raise ValueError("Expected exactly two hidden dimensions.")

        self.activation_name = activation
        self.activation, self.activation_grad = ACTIVATIONS[activation]
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        hidden_dim1, hidden_dim2 = self.hidden_dims

        rng = np.random.default_rng(seed)
        scale_1 = np.sqrt(2.0 / input_dim) if activation == "relu" else np.sqrt(1.0 / input_dim)
        scale_2 = np.sqrt(2.0 / hidden_dim1) if activation == "relu" else np.sqrt(1.0 / hidden_dim1)
        scale_3 = np.sqrt(2.0 / hidden_dim2) if activation == "relu" else np.sqrt(1.0 / hidden_dim2)

        self.W1 = (rng.standard_normal((input_dim, hidden_dim1)) * scale_1).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim1), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_dim1, hidden_dim2)) * scale_2).astype(np.float32)
        self.b2 = np.zeros((1, hidden_dim2), dtype=np.float32)
        self.W3 = (rng.standard_normal((hidden_dim2, num_classes)) * scale_3).astype(np.float32)
        self.b3 = np.zeros((1, num_classes), dtype=np.float32)

    def forward(self, x: np.ndarray) -> ForwardCache:
        z1 = x @ self.W1 + self.b1
        a1 = self.activation(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.activation(z2)
        z3 = a2 @ self.W3 + self.b3
        probs = softmax(z3)
        return ForwardCache(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, probs=probs)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def compute_loss_and_grads(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight_decay: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        cache = self.forward(x)
        batch_size = x.shape[0]

        one_hot = np.zeros_like(cache.probs)
        one_hot[np.arange(batch_size), y] = 1.0

        ce_loss = -np.log(np.clip(cache.probs[np.arange(batch_size), y], 1e-12, 1.0)).mean()
        reg_loss = 0.5 * weight_decay * (
            np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2) + np.sum(self.W3 * self.W3)
        )
        total_loss = float(ce_loss + reg_loss)

        dz3 = (cache.probs - one_hot) / batch_size
        dW3 = cache.a2.T @ dz3 + weight_decay * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.activation_grad(cache.z2)
        dW2 = cache.a1.T @ dz2 + weight_decay * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activation_grad(cache.z1)
        dW1 = cache.x.T @ dz1 + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        return total_loss, grads

    def apply_gradients(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        self.W1 -= learning_rate * grads["W1"]
        self.b1 -= learning_rate * grads["b1"]
        self.W2 -= learning_rate * grads["W2"]
        self.b2 -= learning_rate * grads["b2"]
        self.W3 -= learning_rate * grads["W3"]
        self.b3 -= learning_rate * grads["b3"]

    def state_dict(self) -> dict[str, np.ndarray | str]:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
            "activation_name": np.asarray(self.activation_name),
            "hidden_dims": np.asarray(self.hidden_dims, dtype=np.int64),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.W1 = state["W1"].astype(np.float32)
        self.b1 = state["b1"].astype(np.float32)
        self.W2 = state["W2"].astype(np.float32)
        self.b2 = state["b2"].astype(np.float32)
        self.W3 = state["W3"].astype(np.float32)
        self.b3 = state["b3"].astype(np.float32)
        activation_name = str(state["activation_name"])
        self.activation_name = activation_name
        self.activation, self.activation_grad = ACTIVATIONS[activation_name]
        if "hidden_dims" in state:
            self.hidden_dims = tuple(int(dim) for dim in state["hidden_dims"].tolist())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
