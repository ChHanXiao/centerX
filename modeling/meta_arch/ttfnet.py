import math
import numpy as np
import torch
import torch.nn as nn

from ..backbone import build_backbone
from ..losses import KdLoss
from ..losses import GaussianFocalLoss, GIoULoss
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from ..layers import *

__all__ = ["TTFNet"]


@META_ARCH_REGISTRY.register()
class TTFNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.backbone = build_backbone(cfg)
        self.upsample = CenternetDeconv(cfg)
        self.head = TTFnetHead(cfg)
        self.base_loc = None
        self.down_ratio = cfg.MODEL.CENTERNET.DOWN_SCALE
        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        self.loss_heatmap = GaussianFocalLoss(alpha=2.0, gamma=4.0, loss_weight=1)
        self.loss_bbox = GIoULoss(loss_weight=5.0)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        if cfg.MODEL.CENTERNET.KD.ENABLED:
            self.kd_loss = KdLoss(cfg)
            self.kd_without_label = cfg.MODEL.CENTERNET.KD.KD_WITHOUT_LABEL

        self.to(self.device)

    def forward(self, batched_inputs, model_ts=None):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)

        if not self.training:
            # return self.inference(images)
            return self.inference(images, batched_inputs)
        # return self.inference(images, batched_inputs)
        image_shape = images.tensor.shape[-2:]

        features = self.backbone(images.tensor)

        # features = features[self.cfg.MODEL.RESNETS.OUT_FEATURES[0]]
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        loss = {}
        # KD loss
        if model_ts is not None:
            kd_losses = {}
            for idx, model_t in enumerate(model_ts):
                with torch.no_grad():
                    teacher_output = model_t.backbone(images.tensor)
                    teacher_output = model_t.upsample(teacher_output)
                    teacher_output = model_t.head(teacher_output)
                kd_loss = self.kd_loss(pred_dict, teacher_output, idx)  #, model_ts, images)
                kd_losses = {**kd_losses, **kd_loss}
            if self.kd_without_label:
                loss = kd_losses
                return loss
            else:
                loss = {**kd_losses, **loss}

        gt_dict = self.get_ground_truth(batched_inputs, image_shape)
        gt_loss = self.losses(pred_dict, gt_dict)

        loss = {**loss, **gt_loss}
        return loss

    @torch.no_grad()
    def inference(self, images, batched_inputs, K=100):

        features = self.backbone(images.tensor)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        heats, whs = pred_dict['cls'], pred_dict['wh']
        batch, cat, height, width = heats.size()
        batch_bboxes, batch_scores, batch_clses = TFFNetDecoder.decode(heats, whs, self.down_ratio)
        results = []
        for i in range(batch):
            batch_bboxe = batch_bboxes[i]
            batch_score = batch_scores[i]
            batch_clse = batch_clses[i]

            # batch_bboxes, batch_scores, batch_clses = TFFNetDecoder.decode(heat, wh, self.down_ratio)
            bboxes = batch_bboxe.view([-1, 4])
            scores = batch_score.view([-1, 1])
            clses = batch_clse.view([-1, 1])
            idx = scores.argsort(dim=0, descending=True)
            bboxes = bboxes[idx].view([-1, 4])
            scores = scores[idx].view(-1)
            clses = clses[idx].view(-1)

            keepinds = (scores > -0.1)
            bboxes = bboxes[keepinds]
            scores = scores[keepinds]
            labels = clses[keepinds]

            # detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
            # keepinds = (detections[:, -1] > -0.1)
            # detections = detections[keepinds]
            # labels = clses[keepinds]

            scale_x, scale_y = batched_inputs[i]['width'] / float(images.image_sizes[i][1]), \
                               batched_inputs[i]['height'] / float(images.image_sizes[i][0])

            result = Instances(images.image_sizes[i])
            bboxes[:, 0::2] = bboxes[:, 0::2] * scale_x
            bboxes[:, 1::2] = bboxes[:, 1::2] * scale_y

            result.pred_boxes = Boxes(bboxes)
            result.scores = scores
            result.pred_classes = labels
            results.append({"instances": result})

        return results

    def losses(self, pred_dict, gt_dict):
        r"""
        calculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "score_map": gt scoremap,
                "wh": gt width and height of boxes,
                "reg": gt regression of box center point,
                "reg_mask": mask of regression,
                "index": gt index,
            }
            pred(dict): a dict contains all information of prediction
            pred = {
            "cls": predicted score map
            "reg": predcited regression
            "wh": predicted width and height of box
        }
        """

        pred_hm = pred_dict["cls"]
        pred_wh = pred_dict["wh"]
        heatmap = gt_dict["heatmaps"]
        box_target = gt_dict["box_targets"]
        reg_weight = gt_dict["reg_weights"]

        H, W = pred_hm.shape[2:]
        hm_loss = self.loss_heatmap(
            pred_hm,
            heatmap,
            avg_factor=max(1, heatmap.eq(1).sum()))

        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = box_target.permute(0, 2, 3, 1)

        mask = reg_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4
        pos_mask = mask > 0
        weight = mask[pos_mask].float()
        bboxes1 = pred_boxes[pos_mask].view(-1, 4)
        bboxes2 = boxes[pos_mask].view(-1, 4)
        wh_loss = self.loss_bbox(bboxes1, bboxes2, weight, avg_factor=avg_factor)

        loss = {"hm_loss": hm_loss, "wh_loss": wh_loss}
        # print(loss)
        return loss

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs, image_shape):
        return TTFNetGT.generate(self.cfg, batched_inputs, image_shape, device=self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img / 255.) for img in images]
        # images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images = ImageList.from_tensors(images, 32)
        return images


def build_model(cfg):
    model = TTFNet(cfg)
    return model
