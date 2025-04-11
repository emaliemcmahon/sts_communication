# /home/emaliem/miniconda3/envs/moten/bin/python

import argparse
import imageio
import numpy as np
from pathlib import Path
import moten
from tqdm import tqdm
from glob import glob
from skimage.transform import resize
import pandas as pd


class MotionEnergyActivations():
    def __init__(self, args):
        self.process = 'MotionEnergyActivations'
        self.top_dir = args.top_dir
        self.derivatives_dir = f'{self.top_dir}/derivatives'
        self.out_dir = f'{self.derivatives_dir}/{self.process}'
        self.video_dir = f'{self.derivatives_dir}/stimuli'
        self.out_file = f'{self.out_dir}/motion_energy.csv'
        self.fps = 30
        self.reduce_size_prop = 0.25
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def get_moten(self):
        # Create a pyramid of spatio-temporal gabor filters
        videos = pd.read_csv(f'{self.video_dir}/block_videos.csv')
        videos['file_path'] = self.video_dir + '/' + videos['cond_name'] + '/' + videos['video_name']
        files = videos['file_path'].to_list()
        
        out_moten = []
        for file in tqdm(files, desc='Video progress',
                         position=0, leave=True):
            vid_obj = imageio.get_reader(file, 'ffmpeg')
            num_frames=vid_obj.count_frames()
            
            vid = []
            dims = None
            for i in range(int(num_frames)):
                frame = vid_obj.get_data(i).mean(axis=-1)
                if dims is None:
                    dims = tuple([int(i*self.reduce_size_prop) for i in frame.shape])
                resized_frame = resize(frame, dims, anti_aliasing=True)
                vid.append(resized_frame)
            vid = np.array(vid)

            pyramid = moten.get_default_pyramid(vhsize=dims, fps=self.fps)
            moten_features = pyramid.project_stimulus(vid)
            out_moten.append(moten_features.mean()) #average over frames and append to array
        videos['moten'] = out_moten
        df = videos[['video_name', 'cond_name', 'moten']]
        return df

    def run(self):
        df = self.get_moten()
        df.to_csv(self.out_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_dir', '-t', type=str, help='top directory',
                        default='/mindhive/nklab3/users/emaliem/sts_communication')
    args = parser.parse_args()
    MotionEnergyActivations(args).run()

if __name__ == '__main__':
    main()
