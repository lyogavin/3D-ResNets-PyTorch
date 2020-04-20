#!/bin/bash
python3 ./ActionRecognizor.py --n_val_samples 5 --annotation_path ./kinetics_dl/kinetics_700_anno.json --video_jpgs_dir_path ./test_videos/test_v3/ --n_pretrain_classes 700 --pretrain_path ./data/r3d18_K_200ep.pth --model resnet --model_depth 18  --n_classes 700

