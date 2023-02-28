import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import CLIPProcessor, CLIPModel


class TransformerEncoderMapper(nn.Module):
    def __init__(
            self,
            dim_clip: int,
            dim_embedding: int,
            prefix_length: int,
            clip_length: int,
            num_layers: int = 4,
    ):
        super(TransformerEncoderMapper, self).__init__()
        self.dim_clip = dim_clip
        self.dim_embedding = dim_embedding
        self.prefix_length = prefix_length
        self.clip_length = clip_length
        self.num_layers = num_layers

        self.encoded = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.dim_embedding,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=self.num_layers,
        )
        self.linear = nn.Linear(
            self.dim_clip,
            self.clip_length * self.dim_embedding
        )
        self.prefix_const = nn.Parameter(
            torch.randn(self.prefix_length, self.dim_embedding),
            requires_grad=True
        )

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0],
            *self.prefix_const.shape
        )
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out  # (batch, prefix_length, dim_embedding)


class TagDecoder(nn.Module):
    def __init__(
            self,
            dim_embedding: int,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            nhead: int = 4,
            dropout: float = 0.1,
            activation: str = "relu",
    ):
        super(TagDecoder, self).__init__()
        self.dim_embedding = dim_embedding
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.dim_embedding,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=self.num_layers,
        )

    def forward(self, x, tgt, tgt_mask):
        out = self.decoder(
            x,
            tgt,
            tgt_mask=tgt_mask,
        )
        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(
            mask == 0,
            float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        return mask


class TagModel(nn.Module):
    def __init__(
            self,
            dim_clip: int,
            dim_embedding: int,
            prefix_length: int,
            clip_length: int,
            num_layers: int = 4,
    ):
        self.dim_clip = dim_clip
        self.dim_embedding = dim_embedding
        self.prefix_length = prefix_length
        self.clip_length = clip_length
        self.num_layers = num_layers

        self.encoder = TransformerEncoderMapper(
            dim_clip=self.dim_clip,
            dim_embedding=self.dim_embedding,
            prefix_length=self.prefix_length,
            clip_length=self.clip_length,
            num_layers=self.num_layers,
        )
        self.decoder = TagDecoder(
            dim_embedding=self.dim_embedding,
            num_layers=self.num_layers,
        )

    def forward(self, x, tgt):
        tgt_mask = self.decoder.get_tgt_mask(tgt.shape[1])
        x = self.encoder(x)
        out = self.decoder(x, tgt, tgt_mask)
        return out
