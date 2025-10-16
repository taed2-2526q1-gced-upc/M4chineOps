import os
import shutil
import time

import pandas as pd
import random
import cv2

import deepfake_recognition.config as cfg


def sample_split(src_real: str, src_fake: str, dst_dir: str, metadata_dir: str, n_per_class: int = 250):
    """
    Split the dataset into train/val/test sets by sampling the original data.

    Args:
        src_real (str): Path to the original youtube videos.
        src_fake (str): Path to the deepfake videos.
        dst_dir (str): Parent directory for the outputs.
        metadata_dir (str): Directory to save the metadata CSV files.

        n_per_class (int): Number of samples to draw from each class.
    """

    splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    real_videos = [f for f in os.listdir(src_real) if f.endswith('.mp4')]
    fake_videos = [f for f in os.listdir(src_fake) if f.endswith('.mp4')]

    # sample equally from both classes 
    random.seed(42)
    real_sample = random.sample(real_videos, min(n_per_class, len(real_videos)))
    fake_sample = random.sample(fake_videos, min(n_per_class, len(fake_videos)))

    def make_split(videos, src_dir, label):

        random.shuffle(videos)
        n = len(videos)
        start = 0
        for split, ratio in splits.items():
            print(f'Processing split: {split}...')
            start_t = time.time()

            # create a meta DataFrame for each split
            df = pd.DataFrame(columns = ['filename', 'label', 'frames', 'width', 'height'])
            csv_filename = f'{split}_{label}.csv'
            csv_path = os.path.join(metadata_dir, csv_filename)

            end = start + int(ratio * n)
            split_videos = videos[start:end]
            out_dir = os.path.join(dst_dir, split, label)
            os.makedirs(out_dir, exist_ok=True)
            for v in split_videos:
                full_src_path = os.path.join(src_dir, v)
                cap = cv2.VideoCapture(full_src_path)

                # video properties
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                cap.release()

                new_row = pd.DataFrame([{'filename': v, 'label': label, 
                                         'frames': frames, 'width': width, 'height': height}])
                df = pd.concat([df, new_row], ignore_index=True)
                
                # copy video to output dir
                shutil.copy(os.path.join(src_dir, v), os.path.join(out_dir, v))

            df.to_csv(csv_path, index=False)

            end_t = time.time()
            print(f'Split folder and Metadata CSV created for [{label} - {split}]: {csv_path}.csv with {len(split_videos)} entries in {end_t - start_t:.2f}s!')

            start = end

    # make splits for both real and fake samples
    make_split(real_sample, src_real, 'real')
    make_split(fake_sample, src_fake, 'fake')


def main():
    SRC_REAL = str(cfg.REAL_DATA_SUBDIR)
    SRC_FAKE = str(cfg.FAKE_DATA_SUBDIR)
    SAMPLED_DATA_DIR = str(cfg.SAMPLED_OUTPUT_DIR)

    METADATA_DIR = str(cfg.METADATA_DIR)
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR, exist_ok=True)

    N_PER_CLASS = cfg.N_SAMPLES_PER_CLASS

    sample_split(SRC_REAL, SRC_FAKE, SAMPLED_DATA_DIR, METADATA_DIR, N_PER_CLASS)


if __name__ == "__main__":
    main()
