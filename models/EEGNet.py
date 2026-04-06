import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs, f1=8, d=2, f2=16, temporal_kernel=64, separable_kernel=16):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        temporal_padding = temporal_kernel // 2
        separable_padding = separable_kernel // 2

        # Standard EEGNet-style temporal -> depthwise spatial -> separable temporal blocks.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, temporal_kernel), padding=(0, temporal_padding), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(self.enc_in, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(configs.dropout),
        )

        self.block2_depthwise = nn.Conv2d(
            f1 * d, f1 * d, kernel_size=(1, separable_kernel),
            padding=(0, separable_padding), groups=f1 * d, bias=False
        )
        self.block2_pointwise = nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False)
        self.block2_bn = nn.BatchNorm2d(f2)
        self.block2_act = nn.ELU()
        self.block2_pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.block2_drop = nn.Dropout(configs.dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.enc_in, self.seq_len)
            dummy = self.block1(dummy)
            dummy = self.block2_depthwise(dummy)
            dummy = self.block2_pointwise(dummy)
            dummy = self.block2_bn(dummy)
            dummy = self.block2_act(dummy)
            dummy = self.block2_pool(dummy)
            dummy = self.block2_drop(dummy)
            flat_dim = dummy.flatten(1).shape[1]

        if self.task_name == 'supervised':
            self.projection = nn.Linear(flat_dim, configs.num_class)

    def supervised(self, x_enc, x_mark_enc):
        # Input layout: (batch, timepoints, channels) -> (batch, 1, channels, timepoints)
        x_enc = x_enc.permute(0, 2, 1).unsqueeze(1)
        output = self.block1(x_enc)
        output = self.block2_depthwise(output)
        output = self.block2_pointwise(output)
        output = self.block2_bn(output)
        output = self.block2_act(output)
        output = self.block2_pool(output)
        output = self.block2_drop(output)
        output = output.flatten(1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'supervised':
            return self.supervised(x_enc, x_mark_enc)
        raise ValueError("Task name not recognized or not implemented within the EEGNet Model")
