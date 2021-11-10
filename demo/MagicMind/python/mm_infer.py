import json
import numpy as np
import cv2
import argparse
import sys

import torch

import magicmind.python.runtime as mm
from magicmind.python.runtime import Context

from magicmind_model import MagicMindModel

sys.path.append("/workspace/zhangxiao/work/YOLOX/")
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument("--mm_file_name", help="", type=str, default="yolox_m_int8fp16.model")
    parser.add_argument("--mm_dump", help="", action="store_true")
    parser.add_argument("--image_path", help="", type=str, default="../../../assets/dog.jpg")
    parser.add_argument("--out_path", help="", type=str, default="mm_out.jpg")
    parser.add_argument("--input_shapes", help="", type=list, default=[[1, 3, 640, 640]])
    parser.add_argument("--input_dtypes", help="", type=list, default=["float32"])
    parser.add_argument("--with_p6", action="store_true", help="")

    args = parser.parse_args()

    mm_model = MagicMindModel(args.mm_file_name, mm_dump=args.mm_dump, device_id=0)

    input_shape = args.input_shapes[0][2:4]
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    img = img[np.newaxis, ]
    # img = torch.from_numpy(img)
    outputs = mm_model(img)

    predictions = demo_postprocess(outputs.numpy(), input_shape, p6=args.with_p6)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=0.3, class_names=COCO_CLASSES)

    cv2.imwrite(args.out_path, origin_img)
