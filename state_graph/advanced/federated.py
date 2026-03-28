"""Federated Learning — train across multiple clients without sharing data.

Uses Flower (flwr) framework. Generate server + client code from UI.
"""

from __future__ import annotations

from typing import Any


FL_STRATEGIES = {
    "FedAvg": {"name": "Federated Averaging", "description": "Average model weights across clients (default)"},
    "FedProx": {"name": "FedProx", "description": "FedAvg with proximal term for heterogeneous data"},
    "FedAdam": {"name": "FedAdam", "description": "Server-side Adam optimizer on aggregated updates"},
    "FedYogi": {"name": "FedYogi", "description": "Adaptive server optimizer, handles non-IID data"},
    "FedMedian": {"name": "FedMedian", "description": "Byzantine-robust aggregation using median"},
}


def generate_fl_server(strategy: str = "FedAvg", num_rounds: int = 10, min_clients: int = 2) -> str:
    return f'''"""Federated Learning Server — {strategy}"""
import flwr as fl

strategy = fl.server.strategy.{strategy}(
    min_fit_clients={min_clients},
    min_evaluate_clients={min_clients},
    min_available_clients={min_clients},
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds={num_rounds}),
    strategy=strategy,
)
'''


def generate_fl_client(model_code: str = "") -> str:
    return f'''"""Federated Learning Client"""
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Model (same architecture on every client)
{model_code or "model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))"}
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Local data (each client has its own private data)
# Replace with your data loading
x_local = torch.randn(200, 784)
y_local = torch.randint(0, 10, (200,))
loader = DataLoader(TensorDataset(x_local, y_local), batch_size=32, shuffle=True)


class SGClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [p.detach().numpy() for p in model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(model.parameters(), parameters):
            p.data = torch.tensor(new_p)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        return self.get_parameters(config={{}}), len(loader.dataset), {{}}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader:
                out = model(x)
                total_loss += loss_fn(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        return total_loss / len(loader), total, {{"accuracy": correct / total}}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SGClient())
'''
