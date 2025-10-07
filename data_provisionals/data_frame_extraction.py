import pandas as pd
import os
import cv2
import time

from tqdm import tqdm
from mtcnn import MTCNN

import deepfake_recognition.config as cfg

#################################################
#             Face Frame Extraction             #
# --------------------------------------------- #
# - extracting face frames with MTCNN           #
# - saving the frame list in the dataframe row  #
#################################################


def extract_face_frames_from_video(detector: MTCNN, video_info: dict, split: str, num_frames_to_extract: int):
    """
    Extracts a set number of face frames from a single video and saves them.
    The primary face in each frame is detected using MTCNN and the bounding box around it is saved.

    Args:
        detector (MTCNN): The initialized MTCNN face detector.
        video_info (dict): Dataframe row with all info of the video.
        split (str): Directory to save the extracted face frames.
        num_frames_to_extract (int): The number of frames to sample.

    Returns:
        list: A list of file paths to the saved face frames.
    """
    
    label = video_info['label']
    path_parts = video_info['filepath'].split('/')
    if label == 0:
        filepath = os.path.join(cfg.RAW_DATA_DIR, 'original_sequences', path_parts[0], 'c40/videos', path_parts[1])
    elif label == 1:
        filepath = os.path.join(cfg.RAW_DATA_DIR, 'manipulated_sequences', path_parts[0], 'c40/videos', path_parts[1])

    filename = os.path.splitext(os.path.basename(filepath))[0] # don't get the extension
    
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f'Error: Unable to open video in {filepath}')
        return list()

    frames = video_info['frames']
    step = max(1, frames // num_frames_to_extract) # even intervals

    frame_paths = list()
    for i in tqdm(range(num_frames_to_extract), desc='Extracting Faces'):
        frame_index = i * step
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        _, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame_rgb)
        
        # a face has been detected, save it
        if results:
            x, y, width, height = results[0]['box']
            x, y = max(0, x), max(0, y) 
            face = frame[y : y + height, x : x + width]

            face_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'{split}_data', f'{filename}_FRAME_{i}.jpg')
            cv2.imwrite(face_path, face)
            frame_paths.append(face_path)

    cap.release()    

    return frame_paths


def main():

    detector = MTCNN()
    num_frames_to_extract = 10

    for split in ['train', 'val', 'split']: 
        print(f'Processing {split} split...')
        csv_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'{split}_data.csv')
        df = pd.read_csv(csv_path)

        # create output directory if it doesn't exist
        output_dir = os.path.join(cfg.PROCESSED_DATA_DIR, split)
        os.makedirs(output_dir, exist_ok=True)

        start = time.time()
        df['frame_paths'] = df.apply(lambda row: extract_face_frames_from_video(detector, row, split, num_frames_to_extract), axis=1)
        end = time.time()
        print(f'Video face frames saved sucessfully in {split} folder in {end - start:.2f}s!')

        df.to_csv(csv_path, index=False)
        print(f'Metadata CSV updated successfully: {csv_path} with frame paths!')

if __name__ == '__main__':
    main()