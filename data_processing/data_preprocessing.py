import os
import pandas as pd
import cv2
import time 

from dataclasses import dataclass

import deepfake_recognition.config as cfg

#################################################
#              Folder Reorganizing              #
# --------------------------------------------- #
# - restructuring DVC-tracked files             #
# - creating .csv files with metadata           #
#################################################


@dataclass
class VideoMetadata:
    folderpath: str
    label: int              # 0 for original, 1 for manipulated
    subfolders: list[str]


def preprocess_videos(folder_info: VideoMetadata, metadata_dir: str):
    """
    Preprocess video files in the specified folder by extracting metadata.

    Args:
        folder_info (VideoMetadata): Information about the folder to process.
    """

    for subfolder in folder_info.subfolders:
        print(f'Processing folder: {subfolder} in {folder_info.folderpath}...')
        start = time.time()

        video_folder = os.path.join(folder_info.folderpath, subfolder, 'c40/videos')
        df = pd.DataFrame(columns = ['filepath', 'label', 'frames', 'width', 'height'])
        
        try:
            video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
            
            csv_filename = f'{subfolder}.csv'
            csv_path = os.path.join(metadata_dir, csv_filename)

            for f in video_files:
                f_path = os.path.join(video_folder, f)

                try:
                    cap = cv2.VideoCapture(f_path)

                    # video properties
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    cap.release()

                    new_row = pd.DataFrame([{'filepath': f'{subfolder}/{f}', 'label': folder_info.label, 
                                             'frames': frames, 'width': width, 'height': height}])
                    df = pd.concat([df, new_row], ignore_index=True)
                
                except cv2.error as e:
                    print(f'Error opening video file {f_path}: {e}')
            
            df.to_csv(csv_path, index=False)

            end = time.time()
            print(f'Metadata CSV created successfully: {csv_path} with {len(video_files)} entries in {end - start:.2f}s!')

        except FileNotFoundError as e:
            print(f'Directory {video_folder} not found: {e}')
            continue


def main():

    # populate VideoMetadata instances for original and manipulated videos
    original: VideoMetadata = VideoMetadata(
        folderpath = str(cfg.DATA_DIR) + '/raw/original_sequences',
        label = 0,
        subfolders = ['youtube', 'actors']
    )

    manipulated: VideoMetadata = VideoMetadata(
        folderpath = str(cfg.DATA_DIR) + '/raw/manipulated_sequences',
        label = 1,
        subfolders = ['DeepFakeDetection', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    )

    # create metadata directory if necessary
    metadata_dir = str(cfg.METADATA_DIR)
    if not os.path.exists(metadata_dir):   
        print(f'Creating Metadata directory {metadata_dir}...')
        os.makedirs(metadata_dir)
        print(f'{metadata_dir} created successfully!')
    
    # preprocess original and manipulated videos
    preprocess_videos(original, metadata_dir)
    preprocess_videos(manipulated, metadata_dir)


if __name__ == "__main__":
    main()
