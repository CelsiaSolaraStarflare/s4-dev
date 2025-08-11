"""Convenient S4 model wrapper.

This module exposes a small API around the full S4 implementation used in
``train.py``.  It mirrors the way PyTorch exposes recurrent layers like
``torch.nn.LSTM`` so that users can easily instantiate and stack S4 layers
without digging through the codebase.

The model is backed by :class:`src.models.sequence.backbones.model.SequenceModel`
with the S4 block from :mod:`src.models.sequence.modules.s4block`.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from src.models.sequence.backbones.model import SequenceModel
from src.utils.config import instantiate
from src.utils import registry


class S4Layer(nn.Module):
    """Single S4 layer using the same components as the training pipeline.

    Parameters mirror common RNN layers so it can be used as a drop-in
    replacement for modules like :class:`torch.nn.LSTM`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        bidirectional: bool = False,
        **kernel_args: Any,
    ) -> None:
        super().__init__()
        layer_cfg = {
            "_name_": "s4",
            "layer": "fftconv",
            "kernel": "s4",
            "d_state": d_state,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "transposed": True,
            **kernel_args,
        }
        self.block = instantiate(registry.layer, layer_cfg, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the S4 layer.

        Args:
            x: input tensor of shape ``(batch, length, d_model)``.

        Returns:
            Tensor of the same shape as ``x`` after the S4 transformation.
        """

        # Internal block expects channels-first sequences ``(B, d_model, L)``
        y, _ = self.block(x.transpose(-1, -2))
        return y.transpose(-1, -2)


class S4(nn.Module):
    """Stacked S4 sequence model.

    Parameters are chosen to mirror common recurrent layers.  The network
    processes inputs of shape ``(batch, length, d_model)`` and returns a tuple
    ``(output, state)`` where ``output`` has the same shape as the input.
    """

    def __init__(
        self,
        layers: int,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        bidirectional: bool = False,
        **kernel_args: Any,
    ) -> None:
        super().__init__()

        layer_cfg = {
            "_name_": "s4",
            "layer": "fftconv",
            "bidirectional": bidirectional,
            "kernel": "s4",
            "d_state": d_state,
            **kernel_args,
        }

        self.model = SequenceModel(
            d_model=d_model,
            n_layers=layers,
            dropout=dropout,
            layer=layer_cfg,
            transposed=False,
        )

    def forward(
        self, x: torch.Tensor, state: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """Apply the S4 model.

        Args:
            x: Input tensor of shape ``(batch, length, d_model)``.
            state: Optional recurrent state returned from a previous call.

        Returns:
            A tuple ``(y, next_state)`` where ``y`` has the same shape as ``x``.
        """

        return self.model(x, state=state)

    def step(self, x: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        """Step the model for a single time step.

        This simply forwards to :meth:`SequenceModel.step`.
        """

        return self.model.step(x, state)

    def default_state(self, *batch_shape: int, device=None) -> Any:
        """Return an initial state for the given batch shape."""

        return self.model.default_state(*batch_shape, device=device)


def s4(*args: Any, **kwargs: Any) -> S4:
    """Convenience function returning :class:`S4`.

    This allows users to construct an S4 network with ``s4(layers=2, d_model=128)``
    analogous to calling ``torch.nn.LSTM``.
    """

    return S4(*args, **kwargs)

