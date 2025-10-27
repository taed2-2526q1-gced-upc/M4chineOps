import os

import cv2
from mtcnn import MTCNN
import numpy as np

import deepfake_recognition.config as cfg


def extract_face_frames(filepath: str, detector: MTCNN, IMG_SIZE: tuple[int, int], k: int = 10) -> list[np.ndarray]:
    """
    Extract frames from a video file, at uniform intervals.
    
    Args:
        filepath (str): Path to the video file.
        detector (MTCNN): Face detector instance.
        IMG_SIZE (tuple): Target image size for empty frames.
        k (int): Number of face frames to extract from each video.
    
    Returns:
        list[np.ndarray]: List of extracted face frames as numpy arrays.
    """

    # 1. File loading
    if not os.path.exists(filepath):
        path_parts = filepath.split('/')
        filepath = os.path.join(cfg.RAW_DATA_DIR, 'api_uploads', path_parts[1])
    
    # 2. Frame extraction and face detection
    cap = cv2.VideoCapture(filepath)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // k) # even intervals
    frame_idxs = [i * step for i in range(k)] if k > 1 else [0]

    face_frames = list()
    for curr_frame in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        _, frame = cap.read()

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_face = detector.detect_faces(frame)

            if detected_face: # a face was detected
                x, y, width, height = detected_face[0]['box']
                x, y = max(0, x), max(0, y)

                face_frame = frame[y:y + height, x:x + width]
                face_frames.append(face_frame)
            else:
                print('Appending empty frame due to no face detected.')
                empty_frame = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
                face_frames.append(empty_frame)
        except:
            print('Appending empty frame due to read error.')
            empty_frame = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
            face_frames.append(empty_frame)

    cap.release()

    return face_frames


def extract_and_save_face_paths(detector: MTCNN, filepath: str, filename: str, k: int = 10) -> tuple[list[str], list[str]]:
    """
    Extract face crops from frames in a video file, at uniform intervals.
    
    Args:
        filepath (str): Path to the video file.
        detector (MTCNN): Face detector instance.
        k (int): Number of frames to extract from each video.
    
    Returns:
        boxed_paths: list of paths to saved frames with bounding boxes.
        face_paths: list of paths to saved face crops.
    """
    
    # 1. File loading
    if not os.path.exists(filepath):
        path_parts = filepath.split('/')
        filepath = os.path.join(cfg.RAW_DATA_DIR, 'api_uploads', path_parts[1])

    # 2. Frame extraction and face detection
    cap = cv2.VideoCapture(filepath)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // k) 
    frame_idxs = [i * step for i in range(k)] if k > 1 else [0]

    boxed_paths, face_paths = list(), list()
    for idx, curr_frame in enumerate(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        _, frame = cap.read()

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_face = detector.detect_faces(frame)

            if detected_face: # a face was detected
                x, y, width, height = detected_face[0]['box']
                x, y = max(0, x), max(0, y)

                # original frame + bounding box
                boxed_frame = frame.copy()
                cv2.rectangle(boxed_frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

                boxed_path = os.path.join(cfg.API_UPLOADS_DATA_DIR, f'{filename}_BOX_FRAME_{idx}.jpg')
                cv2.imwrite(boxed_path, cv2.cvtColor(boxed_frame, cv2.COLOR_RGB2BGR))
                boxed_paths.append(boxed_path)

                # detected face crop
                face_frame = frame[y:y + height, x:x + width]
                
                face_path = os.path.join(cfg.API_UPLOADS_DATA_DIR, f'{filename}_FACE_FRAME_{idx}.jpg')
                cv2.imwrite(face_path, cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR))
                face_paths.append(face_path)

                print(f'Face detected and saved in: {face_path}')
                print(f'Frame with bounding box saved in: {boxed_path}')
        except:
            print('Appending empty path due to read error.')
            face_paths.append('')
            boxed_paths.append('')

    cap.release()
    print(f'Total faces detected in {filename}: {len([f for f in face_paths if f != ""])}\n')

    return boxed_paths, face_paths
