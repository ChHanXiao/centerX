import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHead(nn.Module):
    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class TTFnetHead(nn.Module):

    def __init__(self, cfg):
        super(TTFnetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            cfg.MODEL.CENTERNET.NUM_CLASSES,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        self.wh_head = SingleHead(64, 4)
        self.wh_offset_base = cfg.MODEL.CENTERNET.WH_OFFSET_BASE

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = F.relu(self.wh_head(x)) * self.wh_offset_base
        pred = {"cls": cls, "wh": wh}
        return pred
