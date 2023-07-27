# This is a sample Python script.

# Press 鈱僐 to execute it or replace it with your code.
# Press Double 鈬� to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OCOR(nn.Module):
    def __init__(self, hidden_dim=512, kernel_size=3, num_head=4, mlp_1x1=False):
        super(OCOR, self).__init__()

        self.hidden_dim = hidden_dim

        self.num_head = num_head
        self.out_channel = int(num_head * hidden_dim)

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, self.out_channel, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, self.out_channel, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, self.out_channel, kernel_size, padding=padding, bias=False)

        self.conv = nn.Conv2d(
            self.out_channel, hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, self.out_channel, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)

        x = x + virt_feats
        return x
