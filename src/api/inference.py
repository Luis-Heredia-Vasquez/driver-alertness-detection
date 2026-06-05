import os
from typing import Any
import numpy as np


_HAS_TORCH = True
try:
    import torch
    from src.models.cnn import SimpleCNN
except Exception:  # pragma: no cover - import-time
    torch = None
    SimpleCNN = None
    _HAS_TORCH = False


class InferenceEngine:
    """Inference engine with graceful fallback when PyTorch is unavailable.

    If PyTorch is present, uses `SimpleCNN`. Otherwise uses a dummy random
    predictor so the Flask server can start without model dependencies.
    """

    def __init__(self, model_path: str | None = None, device: str = 'cpu') -> None:
        self.device = device
        if _HAS_TORCH and SimpleCNN is not None:
            self._use_torch = True
            self.model = SimpleCNN()
            if model_path and os.path.exists(model_path):
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=device))
                except Exception:
                    # ignore load errors and keep randomly initialized model
                    pass
            self.model.to(device)
            self.model.eval()
        else:
            self._use_torch = False
            self.model = None

    def predict(self, tensor: Any) -> Any:
        """Return predicted class indices.

        Args:
            tensor: either a torch.Tensor or a numpy array-like.
        Returns:
            numpy array of predicted class indices.
        """
        if self._use_torch and torch is not None:
            with torch.no_grad():
                t = tensor
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t).float()
                out = self.model(t.to(self.device))
                return out.argmax(dim=1).cpu().numpy()
        # fallback: return zeros or random prediction with numpy
        arr = np.array(tensor)
        batch = arr.shape[0] if arr.ndim > 0 else 1
        # default single-class 0 predictions
        return np.zeros((batch,), dtype=int)

