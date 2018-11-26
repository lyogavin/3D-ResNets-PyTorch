import sys
import json
import subprocess
from pathlib import Path

if __name__ == '__main__':
    video_dir_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])
    if len(sys.argv) > 3:
        dst_json_path = Path(sys.argv[3])
    else:
        dst_json_path = json_path

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for video_file_path in sorted(video_dir_path.iterdir()):
        file_name = video_file_path.name
        if '.mp4' not in file_name:
            continue
        name = video_file_path.stem

        p = subprocess.Popen(
            'ffprobe {}'.format(video_file_path),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        res = p.communicate()[1].decode('utf-8')

        fps = float([x for x in res.split(',') if 'fps' in x][0].rstrip('fps'))
        json_data['database'][name[2:]]['fps'] = fps

    with open(dst_json_path, 'w') as f:
        json.dump(json_data, f)