import cv2

import sys, os
from tqdm import tqdm

sys.path.insert(0, '.')
from configs import add_centernet_config
from detectron2.config import get_cfg
from inference.ttfnet import build_model
from detectron2.checkpoint import DetectionCheckpointer

if __name__ == "__main__":
    # cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file("yamls/coco/ttfnet_res18_coco_0.5.yaml")

    # model
    model = build_model(cfg)
    DetectionCheckpointer(model).load("exp_results/coco/coco_exp_R18_SGD_0.5/model_final.pth")
    model.eval()

    #txt
    txt = open('byl_det_20201012.txt','w')

    # images
    #lines = open('datasets/bjz_multicameras_20200615/txts/bjz_val_multicameras_20200615.txt').readlines()
    root = './imgs/'
    images = [root + i for i in sorted(os.listdir(root))]
    bs = 8
    for i in tqdm(range(0, len(images), 8)):
        images_rgb = [cv2.imread(j)[:, :, ::-1] for j in images[i:i + 8]]
        img_names = [os.path.basename(j) for j in images[i:i + 8]]
        results = model.inference_on_images(images_rgb, K=100, max_size=640)
        for k, result in enumerate(results):
            cls = result['cls'].cpu().numpy()
            bbox = result['bbox'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            H, W, C = images_rgb[k].shape
            img = images_rgb[k][:, :, ::-1]
            img_name = img_names[k]
            for c, (x1, y1, x2, y2), s in zip(cls, bbox, scores):
                if s < 0.15:
                    continue
                x1 = str(max(0, int(x1)))
                y1 = str(max(0, int(y1)))
                x2 = str(min(W, int(x2)))
                y2 = str(min(H, int(y2)))
                s = str(round(float(s), 3))
                line = ','.join([img_name, s, x1, y1, x2, y2])
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                txt.write(line+'\n')
            cv2.imwrite('datasets/results/'+img_name, img)