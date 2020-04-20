ffmpeg -i ./test_videos/test_v1.mp4 -vf scale=-1:240 -threads 1 ./test_videos/test_v1/image_%05d.jpg

