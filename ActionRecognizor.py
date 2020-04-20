import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import AverageMeter
from opts import parse_opts
from pathlib import Path
import numpy as np



from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from util_scripts.utils import get_n_frames
from datasets.loader import VideoLoader

from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
import logging
import argparse
from mean import get_mean_std


import sys

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

class ActionRecognizor:

    def __init__(self, opt):
        self.opt = opt


    def score(self):

        normalize = get_normalize_method(self.opt.mean, self.opt.std, self.opt.no_mean_norm,
                                         self.opt.no_std_norm)
        spatial_transform = [
            Resize(self.opt.sample_size),
            CenterCrop(self.opt.sample_size),
            ToTensor()
        ]

        spatial_transform.extend([ScaleValue(self.opt.value_scale), normalize])
        spatial_transform = Compose(spatial_transform)

        temporal_transform = []
        if self.opt.sample_t_stride > 1:
            temporal_transform.append(TemporalSubsampling(self.opt.sample_t_stride))
        temporal_transform.append(
            TemporalEvenCrop(self.opt.sample_duration, self.opt.n_val_samples))
        temporal_transform = TemporalCompose(temporal_transform)


        frame_count = get_n_frames(self.opt.video_jpgs_dir_path)

        frame_indices = list(range(0, frame_count))

        frame_indices = temporal_transform(frame_indices)

        spatial_transform.randomize_parameters()

        image_name_formatter = lambda x: f'image_{x:05d}.jpg'

        loader = VideoLoader(image_name_formatter)

        print('frame_indices', frame_indices)
        clip = loader(self.opt.video_jpgs_dir_path, frame_indices[0])


        clips = []

        clip = [spatial_transform(img) for img in clip]
        clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))



        model = generate_model(self.opt)


        model = load_pretrained_model(model, self.opt.pretrain_path, self.opt.model,
                                      self.opt.n_finetune_classes)


        #parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)


        print('clips:', clips)
        outputs = model(torch.unsqueeze(clips[0], 0))
        outputs = F.softmax(outputs, dim=1).cpu()

        print(outputs)

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)



    #parser = argparse.ArgumentParser(description='Start action recognizor...')
    #parser.add_argument('--video_jpgs_dir_path', dest='video_jpgs_dir_path', action='store',
    #                    default='./',
    #                    help='dir path to the path converted images of the video')

    #args = parser.parse_args()
    args = parse_opts()
    args.mean, args.std = get_mean_std(args.value_scale, dataset=args.mean_dataset)
    args.n_input_channels = 3
    if args.pretrain_path is not None:
        args.n_finetune_classes = args.n_classes
        args.n_classes = args.n_pretrain_classes




    recognizor = ActionRecognizor(args)

    recognizor.score()

