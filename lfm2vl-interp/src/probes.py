"""Per-layer linear probes: when does the answer become linearly decodable?

Fits a multinomial logistic regression on the last-token residual stream at each layer to
predict the answer, with a held-out split. Read against the patching curve: probing says
where the information *is present and linearly readable*, patching says where it is
*actually used*. Those are different claims, and the layers where they diverge are the
interesting ones -- information can be decodable long before the model relies on it.

Implemented with plain torch (a few hundred steps of full-batch LBFGS-free gradient
descent) so scikit-learn is not a dependency for one small model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProbeResult:
    layer: int
    train_accuracy: float
    test_accuracy: float
    n_train: int
    n_test: int
    n_classes: int
    majority_baseline: float

    @property
    def above_baseline(self) -> float:
        """Test accuracy minus the majority-class rate -- the only meaningful margin."""
        return self.test_accuracy - self.majority_baseline


def fit_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    layer: int = -1,
    test_fraction: float = 0.3,
    steps: int = 300,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> ProbeResult:
    """Fit one linear probe on ``features`` -> ``labels``.

    Args:
        features: ``(n_samples, hidden)`` residual-stream activations.
        labels: ``(n_samples,)`` integer class ids.
    """
    features = features.float()
    features = (features - features.mean(0)) / (features.std(0) + 1e-6)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=generator)
    n_test = max(1, int(len(features) * test_fraction))
    test_idx, train_idx = order[:n_test], order[n_test:]

    n_classes = int(labels.max()) + 1
    model = torch.nn.Linear(features.shape[1], n_classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    x_train, y_train = features[train_idx], labels[train_idx]
    for _ in range(steps):
        optimiser.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        optimiser.step()

    with torch.no_grad():
        train_accuracy = (model(x_train).argmax(-1) == y_train).float().mean().item()
        x_test, y_test = features[test_idx], labels[test_idx]
        test_accuracy = (model(x_test).argmax(-1) == y_test).float().mean().item()
        # The rate you get by always guessing the most common class in the test split.
        counts = torch.bincount(y_test, minlength=n_classes)
        majority = (counts.max() / counts.sum()).item()

    return ProbeResult(
        layer=layer,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        n_train=len(train_idx),
        n_test=len(test_idx),
        n_classes=n_classes,
        majority_baseline=majority,
    )


def probe_all_layers(
    activations: list[torch.Tensor], labels: torch.Tensor, **kwargs
) -> list[ProbeResult]:
    """Fit one probe per layer. ``activations[layer]`` is ``(n_samples, hidden)``."""
    return [
        fit_probe(features, labels, layer=layer, **kwargs)
        for layer, features in enumerate(activations)
    ]
