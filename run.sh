#!/bin/bash
python -u main.py --root_path ./data --video_path ./HMDB-51/eat/ --result_path results --dataset kinetics  --resume_path ./r3d18_K_200ep.pth --model_depth 18 --n_classes 700 --n_threads 1 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --no_cuda  --annotation_path anno.json
