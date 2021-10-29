import argparse
import os

import torch
import mmcv
import tempfile
import os.path as osp
import torch.distributed as dist
import shutil
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict, get_dist_info
from mmcv.parallel import scatter, collate, MMDataParallel, MMDistributedDataParallel
from mmaction.apis import init_dist
from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['get_logit'] = True
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def multi_gpu_test(model, data_loader, outdir):
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        vid_name = os.path.basename(data['img_path'].data[0][0])
        save_path = os.path.join(outdir, vid_name) + '.pt'

        if not os.path.exists(save_path):
          with torch.no_grad():
              img_group = data['img_group_0'].data[0].cuda()

              bs = img_group.shape[0]
              img_group = img_group.reshape((-1, 3) + img_group.shape[3:])
              # standard protocol i.e. 3 crops * 2 clips
              num_seg = model.module.backbone.nsegments * 2
              # 3 crops to cover full resolution
              num_crops = 3
              img_group = img_group.reshape((num_crops, num_seg) + img_group.shape[1:])

              x1 = img_group[:, ::2, :, :, :]
              x2 = img_group[:, 1::2, :, :, :]
              img_group = torch.cat([x1, x2], 0)
              num_seg = num_seg // 2
              num_clips = img_group.shape[0]
              img_group = img_group.view(num_clips * num_seg, img_group.shape[2], img_group.shape[3], img_group.shape[4])
              feat = model.module.extract_feat(torch.flip(img_group, [-1]))
          
          torch.save(feat[1], save_path)

        if rank == 0:
            batch_size = data['img_group_0'].data[0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoinls'
                                           't file')
    parser.add_argument(
        '--gpus', default=8, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--log', help='output log file')
    parser.add_argument('--fcn_testing', action='store_true', default=False,
                        help='whether to use fcn testing')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='whether to flip videos')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    print('args==>>', args)
    return args


def main():
    args = parse_args()

    assert args.out, ('Please specify the output path for results')

    cfg = mmcv.Config.fromfile(args.config)
    mmcv.mkdir_or_exist(args.out)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if cfg.model.get('necks', None) is not None:
        cfg.model.necks.aux_head_config = None

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8
    if args.fcn_testing:
        cfg.model['cls_head'].update({'fcn_testing': True})
        cfg.model.update({'fcn_testing': True})
    if args.flip:
        cfg.model.update({'flip': True})

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    if not distributed:
        raise NotImplementedError("Non-distributed method not implemented.")
        if args.gpus == 1:
            model = build_recognizer(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            load_checkpoint(model, args.checkpoint, strict=False, map_location='cpu')
            model = MMDataParallel(model, device_ids=[0])

            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                num_gpus=1,
                dist=False,
                shuffle=False)
            outputs = single_test(model, data_loader)
        else:
            model_args = cfg.model.copy()
            model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
            model_type = getattr(recognizers, model_args.pop('type'))

            outputs = parallel_test(
                model_type,
                model_args,
                args.checkpoint,
                dataset,
                _data_func,
                range(args.gpus),
                workers_per_gpu=args.proc_per_gpu)
    else:
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        model = build_recognizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint, strict=False, map_location='cpu')
        model = MMDistributedDataParallel(model.cuda())
        multi_gpu_test(model, data_loader, args.out)


if __name__ == '__main__':
    main()
