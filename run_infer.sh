#!/bin/bash
python3 ./ActionRecognizor.py --video_jpgs_dir_path ./kinetics_dl/abseiling/xmvp3eKN7EM_000000_000010/ --n_pretrain_classes 700 --pretrain_path ./data/r3d18_K_200ep.pth --model resnet --model_depth 18  --n_classes 700

