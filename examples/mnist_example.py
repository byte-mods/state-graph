"""Example: Build and train an MNIST classifier via the StateGraph API.

This demonstrates the programmatic API. You can also do all of this
through the web UI at http://localhost:8765.
"""

import torch

from state_graph.core.engine import TrainingEngine
from state_graph.core.registry import Registry


def main():
    engine = TrainingEngine()

    # Build architecture via the state graph
    engine.graph.add_layer("Linear", {"in_features": 784, "out_features": 256}, activation="ReLU")
    engine.graph.add_layer("BatchNorm1d", {"num_features": 256})
    engine.graph.add_layer("Dropout", {"p": 0.3})
    engine.graph.add_layer("Linear", {"in_features": 256, "out_features": 128}, activation="ReLU")
    engine.graph.add_layer("Dropout", {"p": 0.2})
    engine.graph.add_layer("Linear", {"in_features": 128, "out_features": 10})

    # Configure training
    engine.config.update({
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
    })

    # Register a custom activation formula
    Registry.register_formula_from_string("SquaredReLU", "torch.relu(x) ** 2")

    # Load sample data (simulated MNIST-like)
    x_train = torch.randn(5000, 784)
    y_train = torch.randint(0, 10, (5000,))
    x_val = torch.randn(1000, 784)
    y_val = torch.randint(0, 10, (1000,))

    engine.set_data(x_train, y_train, x_val, y_val)

    # Build and inspect
    result = engine.build()
    print(f"Model built: {result['total_params']} parameters on {result['device']}")
    print(f"\nArchitecture graph:")
    for node in engine.graph.get_sorted_nodes():
        act = f" -> {node.activation}" if node.activation else ""
        print(f"  [{node.position}] {node.layer_type}({node.params}){act}")

    print("\nStarting training...")
    # For the example, we train synchronously
    model = engine.model
    optimizer = engine.optimizer
    loss_fn = engine.loss_fn
    train_loader = engine._train_loader

    for epoch in range(engine.config["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(engine.device)
            y_batch = y_batch.to(engine.device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

            metrics = engine.metrics.collect_step(model, loss.item(), optimizer)

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.1%}")


if __name__ == "__main__":
    main()
