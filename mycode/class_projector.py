import torch
from typing import Callable


# TODO docstrings
class ClassProjector(torch.nn.Module):
    def __init__(self, inner_model: Callable[[torch.Tensor], torch.Tensor], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.inner_model = inner_model


    def select_class(self, X: torch.Tensor) -> None:
        y = self.inner_model(X)
        c = y.argmax(-1)
        batch_idx = torch.arange(len(X), device=X.device)

        self.projection = (batch_idx, c)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = self.inner_model(X)
        y_projected = y[self.projection]

        return y_projected
