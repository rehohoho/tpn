import mmcv
import numpy as np
import os.path as osp
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (GroupImageTransform)
from .utils import to_tensor
import os
import io
import cv2
import h5py
from PIL import Image

from .rawframes_dataset import RawFramesDataset, RawFramesRecord


class RawHdf5Dataset(RawFramesDataset):

    def load_annotations(self, ann_file):
        return [RawFramesRecord(x.strip().split(';')) for x in open(ann_file)]

    def _get_frames(self, record, image_tmpl, modality, indices, skip_offsets):
        if modality not in ['RGB', 'RGBDiff']:
            raise NotImplementedError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff"]')
        
        images = list()
        video_path = osp.join(self.img_prefix, record.path)
        video = h5py.File(video_path, 'r')['video']
        for seg_ind in indices:
            p = int(seg_ind) - 1 # frame index starts from 0 instead of 1
            for i, ind in enumerate(range(0, self.old_length, self.new_step)):
                if p + skip_offsets[i] <= record.num_frames:
                    idx = p + skip_offsets[i]
                else:
                    idx = p
                try:
                    im = Image.open(io.BytesIO(video[idx]))
                    images.extend([np.array(im)])
                except Exception as e:
                    raise Exception(e)
                if p + self.new_step < record.num_frames:
                    p += self.new_step
        
        return images
