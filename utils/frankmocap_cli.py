import os
import sys
import time
from tqdm import tqdm

from frankmocap.demo.demo_options import DemoOptions
import frankmocap.mocap_utils.demo_utils as demo_utils

from utils.frankmocap_utils import FrankMocapWrapper
from utils.handmocap_utils import HandMocapWrapper


if __name__ == "__main__":
    render_video_out = False # deprecated? maybe useful in the future
    dataset = 'video'
    # runner = -1

    if dataset == 'video':
        # assume dataset contains a video path
        video_dir = os.path.expanduser("~/localdata/cam2hand/raw_videos")
        video_names = [
            "output.mp4",
            "output1.mp4",
            "output2.mp4",
        ]
        vid_strs = [f"{video_dir}/{vid}" for vid in video_names]

        is_third_person = False
        outdir_base = os.path.expanduser(f"~/localdata/cam2hand/raw_videos")
    elif dataset == "webcam":
        is_third_person = False
    elif dataset == "realsense":
        is_third_person = False


    if render_video_out:
        for idx, vid_str in enumerate(tqdm(vid_strs)):
            if idx < 0:
                continue

            frankmocap_args = DemoOptions().get_default(video_path=vid_str, is_third_person=is_third_person)

            frankmocap_args.no_video_out = not render_video_out
            frankmocap_args.seq_name = 'clip'

            # hideous hack
            if dataset == 'ego4d':
                relative_dir = vid_str.split('/')[-1].split('.')[0] # alternatively clip_uids[idx] and clip_uids[idx+1]
                next_relative_dir = vid_strs[idx + 1].split('/')[-1].split('.')[0]
            elif dataset == 'vlog':
                relative_dir = vid_str.split('/data/pulkitag/data/dxyang/vlog/')[1].split('/clip.mp4')[0]
                next_relative_dir = vid_strs[idx + 1].split('/data/pulkitag/data/dxyang/vlog/')[1].split('/clip.mp4')[0]
            elif dataset == 'video':
                relative_dir = vid_str.split('/')[-1].split('.mp4')[0]
                next_relative_dir = None
            frankmocap_args.out_dir = f'{outdir_base}/{relative_dir}'
            print(f'hey look here {outdir_base}/{relative_dir}')


            # check if we've already made a video for this one
            expected_video_out_path = f"{outdir_base}/{relative_dir}/clip.mp4"
            if os.path.exists(expected_video_out_path):
                print(f"{expected_video_out_path} already exists; skipping!")
                continue

            # check if the video directory of the next one exists yet so we know that we can turn the current set of frames into a video!
            if idx != len(vid_strs) - 1:
                next_out_dir = f'{outdir_base}/{next_relative_dir}'
                while not os.path.exists(next_out_dir):
                    print(f"waiting for the video processing to catch up!")
                    print(f"currently trying to process {frankmocap_args.out_dir}")
                    print(f"waiting for this to be generated: {next_out_dir}")
                    time.sleep(60)

            demo_utils.gen_video_out(frankmocap_args.out_dir, frankmocap_args.seq_name)
    elif dataset == "webcam":
        print(f"loading mocap wrapper!")
        if is_third_person:
            mocap_engine = FrankMocapWrapper(render_video_out=render_video_out)
        else:
            mocap_engine = HandMocapWrapper(render_video_out=render_video_out)

        mocap_engine.set_new_video_path("webcam")
        mocap_engine.run()
    elif dataset == "realsense":
        print(f"loading mocap wrapper!")
        if is_third_person:
            mocap_engine = FrankMocapWrapper(render_video_out=render_video_out)
        else:
            mocap_engine = HandMocapWrapper(render_video_out=render_video_out)

        mocap_engine.set_new_video_path("realsense")
        mocap_engine.run()
    else:
        print(f"loading mocap wrapper!")
        if is_third_person:
            mocap_engine = FrankMocapWrapper(render_video_out=render_video_out)
        else:
            mocap_engine = HandMocapWrapper(render_video_out=render_video_out)

        print(f"processing videos!")

        for idx, vid_str in enumerate(tqdm(vid_strs)):

            # hideous hack
            if dataset == 'video':
                relative_dir = vid_str.split('/')[-1].split('.mp4')[0]

            expected_video_out_path = f"{outdir_base}/{relative_dir}/clip.mp4"
            print(expected_video_out_path)

            if os.path.exists(expected_video_out_path):
                print(f"{expected_video_out_path} already exists; skipping!")
                continue

            mocap_engine.set_new_video_path(vid_str)
            mocap_engine.run()