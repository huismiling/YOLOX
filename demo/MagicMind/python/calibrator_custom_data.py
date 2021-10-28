from typing import List
import cv2
import numpy

import magicmind.python.runtime as mm
from magicmind.python.common.types import get_numpy_dtype_by_datatype

import os
import sys
sys.path.append("/workspace/zhangxiao/work/YOLOX/")
from yolox.data.data_augment import preproc as preprocess


def load_multi_image(data_paths: List[str], input_wh = List[int], target_dtype: mm.DataType = mm.DataType.FLOAT32) -> numpy.ndarray:
    # Load multiple pre-processed image into a NCHW style ndarray
    images = []
    for path in data_paths:
        img = cv2.imread(path)
        images.append(preprocess(img, input_wh)[0][numpy.newaxis, :])
    ret = numpy.concatenate(tuple(images), axis = 0)
    return numpy.ascontiguousarray(
        ret.astype(dtype = get_numpy_dtype_by_datatype(target_dtype)))


class FixedCalibData(mm.CalibDataInterface):

    def __init__(self, shape: mm.Dims, data_type: mm.DataType, max_samples: int, data_paths: str):
        super().__init__()
        self.shape_ = shape
        self.data_type_ = data_type
        self.batch_size_ = shape.GetDimValue(0)
        self.input_wh = [shape.GetDimValue(3), shape.GetDimValue(2)]
        data_lines = [itd.strip() for itd in open(data_paths).readlines() if os.path.isfile(itd.strip())]
        self.max_samples_ = min(max_samples, len(data_lines))
        self.data_paths_ = data_lines

        self.current_sample_ = None
        self.outputed_sample_count = 0

    def get_shape(self):
        return self.shape_

    def get_data_type(self):
        return self.data_type_

    def get_sample(self):
        return self.current_sample_

    def next(self):
        beg_ind = self.outputed_sample_count
        end_ind = self.outputed_sample_count + self.batch_size_
        if end_ind > self.max_samples_:
            return mm.Status(mm.Code.OUT_OF_RANGE, "End reached")

        self.current_sample_ = load_multi_image(self.data_paths_[beg_ind:end_ind], 
                                                input_wh = self.input_wh, 
                                                target_dtype = self.data_type_)

        self.outputed_sample_count = end_ind
        return mm.Status.OK()

    def reset(self):
        self.current_sample_ = None
        self.outputed_sample_count = 0
        return mm.Status.OK()
