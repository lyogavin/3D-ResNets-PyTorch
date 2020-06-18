#for test:
#python3 -m util_scripts.generate_video_jpgs --n_jobs 5 --fps 1 ../test_kinetics_videos/videos/ ../test_kinetics_videos/jpgs/ kinetics



#for train
nohup python3 -m util_scripts.generate_video_jpgs --n_jobs 7 --fps 1 /media/windows_4/kinetics/compress/train_256/ /media/windows_4/kinetics/compress/train_256_jpgs/ kinetics &


#for val
#nohup python3 -m util_scripts.generate_video_jpgs --n_jobs 7 --fps 1 /media/windows_4/kinetics/compress/val_256/ /media/windows_4/kinetics/compress/val_256_jpgs/ kinetics&
