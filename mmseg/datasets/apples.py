# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import numpy as np
import random

from .builder import DATASETS
from .custom import CustomDataset



@DATASETS.register_module()
class APPLESDataset(CustomDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    CLASSES = ('background', 'plantation')

    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(APPLESDataset, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
