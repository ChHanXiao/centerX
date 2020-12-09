import numpy as np
import torch


class TTFNetGT(object):
    @staticmethod
    def generate(config, batched_input, image_shape, device='cpu'):
        down_ratio = config.MODEL.CENTERNET.DOWN_SCALE
        num_classes = config.MODEL.CENTERNET.NUM_CLASSES
        H, W = image_shape
        output_h, output_w = [int(H / down_ratio), int(W / down_ratio)]
        wh_area_process = config.MODEL.CENTERNET.WH_AREA_PROCESS
        alpha = config.MODEL.CENTERNET.ALPHA

        heatmaps, box_targets, reg_weights = [[] for i in range(3)]
        for data in batched_input:
            # img_size = (data['height'], data['width'])
            bbox_dict = data["instances"].get_fields()

            gt_boxes, gt_labels = bbox_dict["gt_boxes"], bbox_dict["gt_classes"]
            gt_boxes = gt_boxes.tensor.to(device)
            gt_labels = gt_labels.to(device)

            heatmap = gt_boxes.new_zeros((num_classes, output_h, output_w))
            fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
            box_target = gt_boxes.new_ones((4, output_h, output_w))*-1
            reg_weight = gt_boxes.new_zeros((1, output_h, output_w))

            if wh_area_process == 'log':
                boxes_areas_log = TTFNetGT.bbox_areas(gt_boxes).log()
            elif wh_area_process == 'sqrt':
                boxes_areas_log = TTFNetGT.bbox_areas(gt_boxes).sqrt()
            else:
                boxes_areas_log = TTFNetGT.bbox_areas(gt_boxes)
            boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))
            if wh_area_process == 'norm':
                boxes_area_topk_log[:] = 1.

            gt_boxes = gt_boxes[boxes_ind]
            gt_labels = gt_labels[boxes_ind]
            feat_gt_boxes = gt_boxes / down_ratio
            feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                                   max=output_w - 1)
            feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                                   max=output_h - 1)
            feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                                feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

            # we calc the center and ignore area based on the gt-boxes of the origin scale
            # no peak will fall between pixels
            ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                    (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                                   dim=1) / down_ratio).to(torch.int)

            h_radiuses_alpha = (feat_hs / 2. * alpha).int()
            w_radiuses_alpha = (feat_ws / 2. * alpha).int()

            for k in range(boxes_ind.shape[0]):
                cls_id = gt_labels[k]
                fake_heatmap = fake_heatmap.zero_()
                fake_heatmap = TTFNetGT.gen_ellipse_gaussian_target(fake_heatmap, ct_ints[k],
                                                           h_radiuses_alpha[k],
                                                           w_radiuses_alpha[k])
                heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)
                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[0, box_target_inds] = local_heatmap / ct_div


            # tmpp = reg_weight.squeeze(0).cpu()
            # tmp = tmpp.numpy()
            # import matplotlib.pyplot as plt
            # plt.imshow(tmp)
            # plt.axis('off')
            # plt.show()

            heatmaps.append(heatmap)
            box_targets.append(box_target)
            reg_weights.append(reg_weight)

        gt_dict = {
            "heatmaps": torch.stack(heatmaps, dim=0),
            "box_targets": torch.stack(box_targets, dim=0),
            "reg_weights": torch.stack(reg_weights, dim=0),
        }
        return gt_dict

    @staticmethod
    def bbox_areas(bboxes, keep_axis=False):
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (y_max - y_min + 1) * (x_max - x_min + 1)
        if keep_axis:
            return areas[:, None]
        return areas

    @staticmethod
    def gaussian2D_ellipse(radius_h, radius_w, sigma_x=1, sigma_y=1, dtype=torch.float32, device='cpu'):
        """Generate 2D ellipse gaussian kernel.

        Args:
            radius_h (int): Radius of gaussian kernel h.
            radius_w (int): Radius of gaussian kernel w.
            sigma_x (int): Sigma of gaussian function x. Default: 1.
            sigma_y (int): Sigma of gaussian function y. Default: 1.
            dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
            device (str): Device of gaussian tensor. Default: 'cpu'.

        Returns:
            h (Tensor): Gaussian kernel with a
                ``(2 * radius + 1) * (2 * radius + 1)`` shape.
        """
        x = torch.arange(
            -radius_w, radius_w + 1, dtype=dtype, device=device).view(1, -1)
        y = torch.arange(
            -radius_h, radius_h + 1, dtype=dtype, device=device).view(-1, 1)

        h = (-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y))).exp()
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def gen_ellipse_gaussian_target(heatmap, center, radius_h, radius_w, k=1):
        """Generate 2D ellipse gaussian heatmap.

        Args:
            heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
                it and maintain the max value.
            center (list[int]): Coord of gaussian kernel's center.
            radius_h (int): Radius of gaussian kernel h.
            radius_w (int): Radius of gaussian kernel w.
            k (int): Coefficient of gaussian kernel. Default: 1.

        Returns:
            out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
        """
        diameter_h = 2 * radius_h + 1
        diameter_w = 2 * radius_w + 1
        sigma_x = diameter_w / 6
        sigma_y = diameter_h / 6
        gaussian_kernel = TTFNetGT.gaussian2D_ellipse(radius_h, radius_w,
                                             sigma_x=sigma_x, sigma_y=sigma_y,
                                             dtype=heatmap.dtype, device=heatmap.device)
        x, y = center

        height, width = heatmap.shape[:2]

        left, right = min(x, radius_w), min(width - x, radius_w + 1)
        top, bottom = min(y, radius_h), min(height - y, radius_h + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian_kernel[radius_h - top:radius_h + bottom,
                          radius_w - left:radius_w + right]
        out_heatmap = heatmap
        torch.max(
            masked_heatmap,
            masked_gaussian * k,
            out=out_heatmap[y - top:y + bottom, x - left:x + right])

        return out_heatmap

