import torch
import torch.nn.functional as F
from torch import nn

from model.engine_train import all_gather_features


class SupervisedContrasiveLoss(nn.Module):
    """
    Supervised Contrasive Loss. 
    """
    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            features = F.normalize(features, dim=1)           # [B, dim]
        
        sim_mat = torch.matmul(features, features.T) / self.temperature  # [B, B]
        
        labels = labels.contiguous().view(-1, 1)
        mask_same = torch.eq(labels, labels.T).float()
        eye = torch.eye(labels.size(0), device=features.device)
        mask_same -= eye

        exp_logits = torch.exp(sim_mat) * (1 - eye)
        log_prob = sim_mat - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-7)

        num_pos = mask_same.sum(dim=1)
        loss = - (mask_same * log_prob).sum(dim=1) / (num_pos + 1e-7)
        return loss.mean()


class SphereCenterLoss(nn.Module):
    """
    Hypersphere-space center loss for L2-normalized features at spherical surface. 
    The radius r is updated with momentum using IQR-based trimming
    and is clamped to [0, 1] for stability. 
    """

    def __init__(
            self,
            feat_dim: int,
            momentum: float = 0.99,
            normalize: bool = True
        ):
        super().__init__()
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.normalize = normalize

        # Learnable center for the real class
        self.center_real = nn.Parameter(torch.randn(feat_dim))
        
        # Non-learnable radius, updated via momentum
        self.register_buffer('cosine_threshold', torch.tensor(1.0))

    def forward(self, feats):
        if self.normalize:
            feats = F.normalize(feats, dim=1, p=2)
        
        center_norm = F.normalize(self.center_real, dim=0, p=2)

        # Cosine similarity and loss
        cosine_sim = torch.matmul(feats, center_norm)  # (K,)
        loss = (1.0 - cosine_sim).mean()

        # Momentum update with IQR-based trimming
        with torch.no_grad():
            theta = torch.acos(torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7))

            # gather real features from all GPUs
            theta = all_gather_features(theta)

            q1 = torch.quantile(theta, 0.25)
            q3 = torch.quantile(theta, 0.75)
            upper = q3 + 1.5 * (q3 - q1)

            valid = theta[theta <= upper]

            if valid.numel(): 
                update = torch.cos(valid.max())
                new_r = self.momentum * self.cosine_threshold + (1.0 - self.momentum) * update

                self.cosine_threshold.copy_(torch.clamp(new_r, -1.0 + 1e-7, 1.0 - 1e-7))
        
        return loss

