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

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                   float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.old_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _get_frames(self, record, image_tmpl, modality, indices, skip_offsets):
        if modality not in ['RGB', 'RGBDiff']:
            raise NotImplementedError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff"]')
        
        images = list()
        video_path = osp.join(self.img_prefix, record.path)
        video = h5py.File(video_path, 'r')['video']
        print(video_path)
        for seg_ind in indices:
            p = int(seg_ind)
            for i, ind in enumerate(range(0, self.old_length, self.new_step)):
                if p + skip_offsets[i] <= record.num_frames:
                    idx = p + skip_offsets[i]
                else:
                    idx = p
                try:
                    # convert BGR?
                    im = Image.open(io.BytesIO(video[idx])).convert('RGB')
                    images.extend([np.array(im)])
                except Exception as e:
                    print(e)
                if p + self.new_step < record.num_frames:
                    p += self.new_step
        
        return images

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        if self.test_mode:
            segment_indices, skip_offsets = self._get_test_indices(record)
        else:
            segment_indices, skip_offsets = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)

        data = dict(num_modalities=DC(to_tensor(len(self.modalities))),
                    gt_label=DC(to_tensor(record.label), stack=True,
                                pad_dims=None))

        # handle the first modality
        modality = self.modalities[0]
        image_tmpl = self.image_tmpls[0]
        img_group = self._get_frames(
            record, image_tmpl, modality, segment_indices, skip_offsets)

        flip = True if np.random.rand() < self.flip_ratio else False
        if (self.img_scale_dict is not None
                and record.path in self.img_scale_dict):
            img_scale = self.img_scale_dict[record.path]
        else:
            img_scale = self.img_scale
        (img_group, img_shape, pad_shape,
         scale_factor, crop_quadruple) = self.img_group_transform(
            img_group, img_scale,
            crop_history=None,
            flip=flip, keep_ratio=self.resize_keep_ratio,
            div_255=self.div_255,
            is_flow=True if modality == 'Flow' else False)
        ori_shape = (256, 340, 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            crop_quadruple=crop_quadruple,
            flip=flip)
        # [M x C x H x W]
        # M = 1 * N_oversample * N_seg * L
        if self.input_format == "NCTHW":
            img_group = img_group.reshape(
                (-1, self.num_segments, self.new_length) + img_group.shape[1:])
            # N_over x N_seg x L x C x H x W
            img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
            # N_over x N_seg x C x L x H x W
            img_group = img_group.reshape((-1,) + img_group.shape[2:])
            # M' x C x L x H x W
        data.update(dict(
            img_group_0=DC(to_tensor(img_group), stack=True, pad_dims=2),
            img_meta=DC(img_meta, cpu_only=True),
            img_path=DC(record.path, cpu_only=True),
            over_sample=DC(self.oversample, cpu_only=True),
        ))

        return data
