import torch
from typing import Callable


# TODO docstrings
class ClassProjector(torch.nn.Module):
    def __init__(self, inner_model: Callable[[torch.Tensor], torch.Tensor], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.inner_model = inner_model


    def check_init(self) -> None:
        """Check that self.projection has been initialized, i.e. that "select_class" was called. Function is trivially overwritten in "select_class"."""
        raise RuntimeError("self.projection not initialized. Make sure to call 'select_class' before invoking the forward pass.")


    def select_class(self, X: torch.Tensor) -> None:
        y = self.inner_model(X)
        c = y.argmax(-1)
        batch_idx = torch.arange(len(X), device=X.device)

        self.projection = (batch_idx, c)
        self.check_init = lambda: None


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.check_init()

        y = self.inner_model(X)
        y_projected = y[self.projection]

        return y_projected
