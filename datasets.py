import numpy as np
import torch
from torch.utils.data import Dataset


class SingleInputDataset(Dataset):
    """For models that take one embedding vector: image-only, text-only, or prompt-only."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class FusionDataset(Dataset):
    """For fusion models that take two embedding vectors: image + text/prompt."""

    def __init__(self, Xi: np.ndarray, Xt: np.ndarray, y: np.ndarray):
        self.Xi = torch.from_numpy(Xi).float()
        self.Xt = torch.from_numpy(Xt).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.Xi[i], self.Xt[i]), self.y[i]
