import subprocess
import argparse
from pathlib import Path
import logging
from tqdm import tqdm


import sys


from joblib import Parallel, delayed

logger = logging.getLogger()
formatter = logging.Formatter(
    '%(process)d-%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]')
file_handler = logging.FileHandler("./generate_video_jpgs.log")
handler = logging.StreamHandler(sys.stdout)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    #if ext != video_file_path.suffix:
    #    logger.error(f"unknown ext: {ext}")
    #    return False


    #logger.info(f"suffix: {video_file_path.suffix}")

    try:
    if video_file_path.suffix == ".mkv" or video_file_path.suffix == ".webm":
        ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                       '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                       'stream=width,height,avg_frame_rate -show_entries format=duration').split()
    else:

        ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                       '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                       'stream=width,height,avg_frame_rate,duration').split()
    ffprobe_cmd.append(str(video_file_path))
    logger.info("running: %s" % " ".join(ffprobe_cmd))
    ffoutput = None

    try:
        ffoutput = subprocess.check_output(' '.join(ffprobe_cmd), shell=True, stderr=subprocess.STDOUT)
        #p = subprocess.run(ffprobe_cmd, capture_output=True)
        #res = p.stdout.decode('utf-8').splitlines()
        res = ffoutput.decode('utf-8').splitlines()
        if len(res) < 4:
            logger.error(f"error ffprobe, output len less then 4: {ffoutput}")
            return False

    except subprocess.CalledProcessError as err:
        logger.error(f"error ffprobe exp:{err}, output:{ffoutput}")
        return False

    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
    duration = float(res[3])

    if fps > 0:
        frame_rate = fps

    n_frames = int(frame_rate * duration)

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    logger.info(f"exist:{n_exist_frames}  expected:{n_frames}")
    if n_exist_frames >= n_frames:
        logger.info(f"{n_exist_frames} already exists more than expected:{n_frames}, return")
        return True

    width = int(res[0])
    height = int(res[1])

    if width > height:
        vf_param = 'scale=-1:{}'.format(size)
    else:
        vf_param = 'scale={}:-1'.format(size)

    if fps > 0:
        vf_param += ',minterpolate={}'.format(fps)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    logger.info(f"to run:{ffmpeg_cmd}")
    #subprocess.run(ffmpeg_cmd)
    ffoutput = None
    try:
        ffoutput = subprocess.check_output(' '.join(ffmpeg_cmd), shell=True, stderr=subprocess.STDOUT)

    except subprocess.CalledProcessError as err:
        logger.error(f"error ffprobe exp:{err}, output:{ffoutput}")
        return False

    except BaseException as error:
        logger.info('An exception occurred: {}'.format(error))

    return True


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    processed_files = 0
    for video_file_path in tqdm(sorted(class_dir_path.iterdir()), postfix=f"[class:{class_dir_path}]"):
        logger.info(f"processing: {video_file_path}")
        res = video_process(video_file_path, dst_class_path, ext, fps, size)
        if not res:
            logger.error(f"processing {video_file_path} failed")
            return
        processed_files += 1
    logger.info(f"processed {processed_files} videos for {class_dir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        'dataset',
        default='',
        type=str,
        help='Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet)')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=-1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    parser.add_argument(
        '--size', default=240, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    if args.dataset in ['kinetics', 'mit', 'activitynet']:
        ext = '.mp4'
    else:
        ext = '.avi'

    if args.dataset == 'activitynet':
        video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
        status_list = Parallel(
            n_jobs=args.n_jobs,
            verbose=10,
            backend='threading')(delayed(video_process)(
                video_file_path, args.dst_path, ext, args.fps, args.size)
                                 for video_file_path in video_file_paths)
    else:
        class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
        test_set_video_path = args.dir_path / 'test'
        if test_set_video_path.exists():
            class_dir_paths.append(test_set_video_path)

        status_list = Parallel(
            n_jobs=args.n_jobs,
            backend='threading')(delayed(class_process)(
                class_dir_path, args.dst_path, ext, args.fps, args.size)
                                 for class_dir_path in tqdm(class_dir_paths, postfix="[overall]"))

        logger.info(f"status: {status_list}")
