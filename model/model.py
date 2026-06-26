import torch
import torch.nn as nn
from timm import create_model
from timm.layers import Mlp


class TwinNeXt(nn.Module): 
    """ 
    TwinNeXt: two independent ConvNeXt backbones (from timm with pretrained weights)
    extract features from two images, concatenate the features,
    then pass through an MLP to obtain the final embeddings. 
    """
    def __init__(
            self,
            backbone_name: str = "convnext_small",
            mlp_hidden_dims: int = 512,
            out_dim: int = 128
        ): 
        """
        Args:
            backbone_name (str): Model name for ConvNeXt with pretrained weights. 
            mlp_hidden_dims (list[int]): Hidden dimensions of the MLP. 
            out_dim (int): Final feature dimension.
        """
        super(TwinNeXt, self).__init__()

        # Two independent ConvNeXtv2 backbones with pretrained weights
        self.backbone1 = create_model(backbone_name, pretrained=True, num_classes=0) # feature dimension is 1024
        self.backbone2 = create_model(backbone_name, pretrained=True, num_classes=0)

        # Build MLP using timm's Mlp
        self.mlp = Mlp(
            in_features=self.backbone1.num_features + self.backbone2.num_features, 
            hidden_features=mlp_hidden_dims, 
            out_features=out_dim
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x1 (torch.Tensor): Local slice tensor (B, 3, H, W).
            x2 (torch.Tensor): Global image tensor (B, 3, H, W).
        
        Returns: 
            torch.Tensor: Output feature tensor (B, out_dim). 
        """
        f1 = self.backbone1(x1)
        f2 = self.backbone2(x2)
        feat = torch.cat([f1, f2], dim=1)
        out = self.mlp(feat)
        return out

