import json
import numpy as np
import cv2
import argparse
import sys
from demo.MagicMind.python.magicmind_model import MagicMindModel

import magicmind.python.runtime as mm
from magicmind.python.runtime import Context

from magicmind_model import MagicMindModel

sys.path.append("/workspace/zhangxiao/work/YOLOX/")
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis

# 生成模型
def build_model(args):
    model = mm.Model()
    model.deserialize_from_file(args.mm_file_name)
    assert model != None, "Failed to build model"
    return model

# 创建推理模型所需的上下文
def create_context(model, args):
    econfig = mm.Model.EngineConfig()
    econfig.device_type = "MLU"
    engine = model.create_i_engine(econfig)
    assert engine != None, "Failed to create engine"
    context = engine.create_i_context()
    
    if args.mm_dump:
        dumpinfo = Context.ContextDumpInfo(path="/tmp/output_pb/", tensor_name=[], dump_mode=1, file_format=0)
        # dumpinfo.val.dump_mode = 1   # -1 关闭dump 模式; 0 dump 指定tensor; 1 dump所有tensor; 2 dump 输出tensor
        # dumpinfo.val.path = "/tmp/output"# 将dump 结果存放到/tmp/output ⽬录下
        # dumpinfo.val.tensor_name = [] # dump 所有tensor 信息
        # dumpinfo.val.file_format = 0 # 0 ⽂件保存为pb; 1 ⽂件保存为pbtxt
        context.set_context_dump_info(dumpinfo)

    return context

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基础参数，必选
    parser.add_argument("--builder_config", help="",type=str, default="builder_config.json")
    parser.add_argument("--mm_file_name", help="", type=str, default="yolox_m_quantized.model")
    parser.add_argument("--mm_dump", help="", action="store_true")
    parser.add_argument("--image_path", help="", type=str, default="../../../assets/dog.jpg")
    parser.add_argument("--input_shapes", help="", type=list, default=[[1, 3, 640, 640]])
    parser.add_argument("--input_dtypes", help="", type=list, default=["float32"])
    parser.add_argument("--with_p6", action="store_true", help="")

    args = parser.parse_args()

    mm_model = MagicMindModel(args.mm_file_name, mm_dump=args.mm_dump, device_id=0)

    input_shape = args.input_shapes[0][2:4]
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    outputs = mm_model(img[np.newaxis, ])

    predictions = demo_postprocess(outputs[0].asnumpy(), input_shape, p6=args.with_p6)[0]

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

    cv2.imwrite("mm_out.jpg", origin_img)
