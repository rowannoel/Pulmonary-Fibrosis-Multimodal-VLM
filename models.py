import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """Simple MLP: input -> hidden -> logits(2)."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class ProfFusionModel(nn.Module):
    """
    Fusion model:
      image embedding -> projection -> L2 normalization
      text/prompt embedding -> projection -> L2 normalization
      concatenate -> MLP -> logits
    """

    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        d: int = 512,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_img = nn.Linear(img_dim, d)
        self.proj_txt = nn.Linear(txt_dim, d)
        self.clf = MLPClassifier(input_dim=2 * d, hidden_dim=hidden, dropout=dropout)

    def forward(self, e_img, e_txt):
        p_img = self.proj_img(e_img)
        p_txt = self.proj_txt(e_txt)

        z_img = F.normalize(p_img, p=2, dim=1)
        z_txt = F.normalize(p_txt, p=2, dim=1)

        x = torch.cat([z_img, z_txt], dim=1)
        return self.clf(x)
