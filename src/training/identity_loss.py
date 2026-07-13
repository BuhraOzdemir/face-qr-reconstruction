"""
Identity Loss — training-only module.

Uses a frozen InceptionResnetV1 (facenet-pytorch, pretrained on VGGFace2)
to extract face embeddings from generated and ground-truth images, then
computes cosine distance as the identity loss.

Loss = 1 - cosine_similarity(emb_generated, emb_target)

This model is NEVER exported or used during inference / mobile deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class IdentityLoss(nn.Module):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # Pure PyTorch face recognition model — fully differentiable
        self.resnet = InceptionResnetV1(pretrained="vggface2").to(device)

        # Eval mode: disables dropout and BN stat updates
        self.resnet.eval()

        # Freeze all parameters — identity model is never optimized
        for param in self.resnet.parameters():
            param.requires_grad_(False)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Prepare images for InceptionResnetV1:
          - Input : [0, 1]
          - Resize to 160x160 (model's required input size)
          - Rescale to [-1, 1] as expected by InceptionResnetV1
        """
        images = images * 2.0 - 1.0
        images = F.interpolate(
            images,
            size=(160, 160),
            mode="bilinear",
            align_corners=False,
        )
        return images

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        generated : (B, 3, H, W) in [0, 1] — converted by trainer before call
        target    : (B, 3, H, W) in [0, 1] — converted by trainer before call

        Gradient flows only through generated embeddings;
        target embeddings are detached (no need to backprop through GT).
        """
        gen_input = self._preprocess(generated)
        tgt_input = self._preprocess(target)

        # Embedding extraction — gradient flows through generated path
        emb_gen = self.resnet(gen_input)              # (B, 512), grad-enabled
        with torch.no_grad():
            emb_tgt = self.resnet(tgt_input)          # (B, 512), detached

        cos_sim = F.cosine_similarity(emb_gen, emb_tgt, dim=1)
        return (1.0 - cos_sim).mean()
