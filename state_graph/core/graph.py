"""State graph representation of a neural network architecture.

Tracks the model as a directed graph of layers, enabling real-time
visualization and dynamic architecture modification.

Supports:
- Sequential chains (nn.Sequential)
- Branching architectures (skip connections, residuals, multi-path)
- Multi-input / multi-output models
- Custom forward logic via node groups and merge operations
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from state_graph.core.registry import Registry


@dataclass
class LayerNode:
    """A single node in the state graph representing a layer."""

    id: str
    layer_type: str
    params: dict[str, Any]
    activation: str | None = None
    position: int = 0  # Order in the sequential graph
    group: str = "main"  # Node group: "main", "encoder", "decoder", "discriminator", etc.
    # Branching support
    inputs: list[str] | None = None  # Node IDs this node reads from (None = previous in sequence)
    merge_mode: str | None = None  # "add", "concat", "multiply", "gate" — how to merge multiple inputs

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "layer_type": self.layer_type,
            "params": self.params,
            "activation": self.activation,
            "position": self.position,
            "group": self.group,
        }
        if self.inputs:
            d["inputs"] = self.inputs
        if self.merge_mode:
            d["merge_mode"] = self.merge_mode
        return d


@dataclass
class Edge:
    """Connection between two layer nodes."""

    source: str
    target: str

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target}


class StateGraph:
    """Directed graph representing the model architecture.

    Supports dynamic modification - add/remove/reorder layers at any time,
    then rebuild the PyTorch model from the graph.

    Build modes:
    - "sequential" (default): nn.Sequential from sorted nodes
    - "branching": DAG with skip connections and multi-path
    """

    def __init__(self) -> None:
        self.nodes: dict[str, LayerNode] = {}
        self.edges: list[Edge] = []
        self._model: nn.Module | None = None

    def add_layer(
        self,
        layer_type: str,
        params: dict[str, Any] | None = None,
        activation: str | None = None,
        position: int | None = None,
        group: str = "main",
        inputs: list[str] | None = None,
        merge_mode: str | None = None,
    ) -> str:
        """Add a layer to the graph. Returns the node ID."""
        node_id = str(uuid.uuid4())[:8]
        if params is None:
            params = {}
        if position is None:
            position = len(self.nodes)

        # Shift existing nodes at or after this position
        for node in self.nodes.values():
            if node.position >= position:
                node.position += 1

        self.nodes[node_id] = LayerNode(
            id=node_id,
            layer_type=layer_type,
            params=params,
            activation=activation,
            position=position,
            group=group,
            inputs=inputs,
            merge_mode=merge_mode,
        )
        self._rebuild_edges()
        self._model = None  # Invalidate cached model
        return node_id

    def remove_layer(self, node_id: str) -> None:
        """Remove a layer from the graph."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")
        removed_pos = self.nodes[node_id].position
        del self.nodes[node_id]

        # Shift positions down
        for node in self.nodes.values():
            if node.position > removed_pos:
                node.position -= 1
            # Clean up references
            if node.inputs:
                node.inputs = [i for i in node.inputs if i != node_id]
                if not node.inputs:
                    node.inputs = None

        self._rebuild_edges()
        self._model = None

    def update_layer(
        self,
        node_id: str,
        layer_type: str | None = None,
        params: dict[str, Any] | None = None,
        activation: str | None = None,
        inputs: list[str] | None = None,
        merge_mode: str | None = None,
    ) -> None:
        """Update an existing layer's configuration."""
        node = self.nodes[node_id]
        if layer_type is not None:
            node.layer_type = layer_type
        if params is not None:
            node.params = params
        if activation is not None:
            node.activation = activation
        if inputs is not None:
            node.inputs = inputs if inputs else None
        if merge_mode is not None:
            node.merge_mode = merge_mode
        self._model = None

    def add_skip_connection(self, from_id: str, to_id: str, merge_mode: str = "add") -> None:
        """Add a skip/residual connection from one node to another."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise KeyError("Invalid node IDs")
        target = self.nodes[to_id]
        if target.inputs is None:
            target.inputs = []
        if from_id not in target.inputs:
            target.inputs.append(from_id)
        target.merge_mode = merge_mode
        self._rebuild_edges()
        self._model = None

    def reorder_layer(self, node_id: str, new_position: int) -> None:
        """Move a layer to a new position."""
        node = self.nodes[node_id]
        old_pos = node.position

        if old_pos == new_position:
            return

        for other in self.nodes.values():
            if other.id == node_id:
                continue
            if old_pos < new_position:
                if old_pos < other.position <= new_position:
                    other.position -= 1
            else:
                if new_position <= other.position < old_pos:
                    other.position += 1

        node.position = new_position
        self._rebuild_edges()
        self._model = None

    def _rebuild_edges(self) -> None:
        """Rebuild edges based on position ordering + explicit inputs."""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.position)
        self.edges = []

        for i in range(len(sorted_nodes) - 1):
            self.edges.append(Edge(sorted_nodes[i].id, sorted_nodes[i + 1].id))

        # Add explicit skip connection edges
        for node in self.nodes.values():
            if node.inputs:
                for src_id in node.inputs:
                    if src_id in self.nodes:
                        self.edges.append(Edge(src_id, node.id))

    def _has_branching(self) -> bool:
        """Check if any node has explicit inputs (skip connections)."""
        return any(n.inputs for n in self.nodes.values())

    def build_model(self) -> nn.Module:
        """Build a PyTorch model from the graph.

        If no branching: returns nn.Sequential (backward compatible).
        If branching: returns a BranchingModel with DAG execution.
        """
        if self._has_branching():
            return self._build_branching_model()
        return self._build_sequential_model()

    def _build_sequential_model(self) -> nn.Module:
        """Build a simple nn.Sequential model."""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.position)
        modules: list[nn.Module] = []

        for node in sorted_nodes:
            layer_cls = Registry.get_layer(node.layer_type)
            layer = layer_cls(**node.params)
            modules.append(layer)

            if node.activation:
                act_cls = Registry.get_activation(node.activation)
                modules.append(act_cls())

        self._model = nn.Sequential(*modules)
        return self._model

    def _build_branching_model(self) -> nn.Module:
        """Build a model with skip connections and multi-path routing."""
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.position)

        # Build each layer module
        node_modules = {}
        for node in sorted_nodes:
            layer_cls = Registry.get_layer(node.layer_type)
            layer = layer_cls(**node.params)
            if node.activation:
                act_cls = Registry.get_activation(node.activation)
                layer = nn.Sequential(layer, act_cls())
            node_modules[node.id] = layer

        # Create the branching model
        model = _BranchingModel(sorted_nodes, node_modules)
        self._model = model
        return self._model

    def get_model(self) -> nn.Module:
        """Get the current model, building it if needed."""
        if self._model is None:
            return self.build_model()
        return self._model

    def get_sorted_nodes(self) -> list[LayerNode]:
        return sorted(self.nodes.values(), key=lambda n: n.position)

    def to_dict(self) -> dict:
        """Serialize the graph for the UI."""
        sorted_nodes = self.get_sorted_nodes()
        return {
            "nodes": [n.to_dict() for n in sorted_nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def get_param_count(self) -> dict:
        """Get parameter counts per layer."""
        model = self.get_model()

        if isinstance(model, _BranchingModel):
            counts = {}
            for node_id, module in model.node_modules.items():
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                counts[node_id] = {
                    "total": total,
                    "trainable": trainable,
                    "shape": {name: list(p.shape) for name, p in module.named_parameters()},
                }
            return counts

        # Sequential model
        counts = {}
        sorted_nodes = self.get_sorted_nodes()
        module_idx = 0
        for node in sorted_nodes:
            if module_idx < len(model):
                m = model[module_idx]
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                counts[node.id] = {
                    "total": total,
                    "trainable": trainable,
                    "shape": {name: list(p.shape) for name, p in m.named_parameters()},
                }
                module_idx += 1
                if node.activation:
                    module_idx += 1

        return counts


class _BranchingModel(nn.Module):
    """Model with DAG execution — supports skip connections, residuals, multi-path."""

    def __init__(self, sorted_nodes: list[LayerNode], node_modules: dict[str, nn.Module]):
        super().__init__()
        self.node_order = [n.id for n in sorted_nodes]
        self.node_inputs = {n.id: n.inputs for n in sorted_nodes}
        self.node_merge_modes = {n.id: n.merge_mode for n in sorted_nodes}

        # Register as ModuleDict for proper parameter tracking
        self.node_modules = nn.ModuleDict(node_modules)

        # Merge projection layers for concat mode
        self.merge_projs = nn.ModuleDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations: dict[str, torch.Tensor] = {}
        current = x

        for node_id in self.node_order:
            module = self.node_modules[node_id]
            inputs = self.node_inputs.get(node_id)
            merge_mode = self.node_merge_modes.get(node_id)

            # Compute this node's output
            output = module(current)

            # Merge with skip connection inputs if specified
            if inputs:
                for src_id in inputs:
                    if src_id in activations:
                        src_out = activations[src_id]
                        if merge_mode == "add":
                            # Ensure shapes match for residual add
                            if output.shape == src_out.shape:
                                output = output + src_out
                        elif merge_mode == "multiply":
                            if output.shape == src_out.shape:
                                output = output * src_out
                        elif merge_mode == "concat":
                            output = torch.cat([output, src_out], dim=-1)
                        elif merge_mode == "gate":
                            if output.shape == src_out.shape:
                                gate = torch.sigmoid(src_out)
                                output = output * gate

            activations[node_id] = output
            current = output

        return current
