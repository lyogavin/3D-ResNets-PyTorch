#!/bin/bash
python3 ./ActionRecognizor.py --n_val_samples 30  --annotation_path ./kinetics_dl/kinetics_700_anno.json --video_jpgs_dir_path ./test_videos/test_v1/ --n_pretrain_classes 700 --pretrain_path ./data/r3d101_K_200ep.pth --model resnet --model_depth 101  --n_classes 700

