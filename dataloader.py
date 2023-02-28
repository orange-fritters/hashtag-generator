import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from transformers import CLIPProcessor, CLIPModel
import skimage.io as io
from PIL import Image


class HarrisonDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            prefix_length: int,
            normalize_prefix: bool = False,
    ):
        self.data = pd.read_csv(data_path)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.prefix_length = prefix_length
        self.tokenizer = self.processor.tokenizer
        self.tags = self.data['tags']
        self.tokens = [torch.tensor(self.tokenizer.encode(
            tag), dtype=torch.int64) for tag in self.tags]
        self.max_seq_length = max([len(token) for token in self.tokens])
        self.normalize_prefix = normalize_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # token and mask part
        tokens, mask = self.pad_tokens(idx)

        # image embed part
        img_dir = self.data['img_dir'][idx]
        image = Image.open(img_dir).convert("RGB")
        pixel_values = self.processor.feature_extractor(
            image, return_tensors="pt").pixel_values
        embed = self.clip.get_image_features(pixel_values)
        if self.normalize_prefix:
            embed = embed.float()
            embed = embed / embed.norm(p=2, dim=-1)
        return tokens, mask, embed

    def pad_tokens(
            self,
            index: int):
        tokens = self.tokens[index]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat(
                (tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.tokens[index] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[index] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat(
            (torch.ones(self.prefix_length), mask),
            dim=0
        )  # adding prefix mask
        return tokens, mask


if __name__ == "__main__":
    dataset = HarrisonDataset(
        data_path="/content/harrison/HARRISON/data.csv", prefix_length=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
