import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from codecarbon import EmissionsTracker

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

import deepfake_recognition.config as cfg


def extract_video_frames_uniform(filepath: str, IMG_SIZE: tuple[int, int], k: int = 10) -> list[np.ndarray]:
    """
    Extract frames from a video file, at uniform intervals.
    
    Args:
        filepath (str): Path to the video file.
        IMG_SIZE (tuple): Target image size for empty frames.
        k (int): Number of frames to extract from each video.
    
    Returns:
        list[np.ndarray]: List of extracted frames as numpy arrays.
    """

    cap = cv2.VideoCapture(filepath)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // k) # even intervals
    frame_idxs = [i * step for i in range(k)] if k > 1 else [0]

    frames = list()
    for curr_frame in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        _, frame = cap.read()

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        except:
            print('Appending empty frame due to read error.')
            empty_frame = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
            frames.append(empty_frame)

    cap.release()

    return frames


def preprocess_for_Xception(frames: np.ndarray, img_size: tuple[int, int] = (299, 299)) -> np.ndarray:
    """
    Preprocess frames per Xception. 
    Return np.array [N, H, W, 3] ready for model input.

    Args:
        frames (np.ndarray): List of frames as numpy arrays.
        img_size (tuple): Target size for resizing frames.
    
    Returns:
        np.ndarray: Preprocessed frames array.
    """

    arr = list()
    for f in frames:
        f = cv2.resize(f, img_size, interpolation = cv2.INTER_AREA)
        arr.append(f.astype(np.float32))
    x = np.stack(arr, axis=0)            
    x = preprocess_input(x) # Xception preprocess
    
    return x


def build_frame_embeddings(emb_model: Model, frames_np: np.ndarray) -> np.ndarray:
    """
    Computes k embeddings for each video, using Kera's Xception model.
    Aggregates them into a single video embedding.

    Args:
        emb_model (Model): Kera's Xception model for embeddings.
        frames_np (np.ndarray): Array of preprocessed frames [k, H, W, 3].
        agg_method (str): 'mean' or 'sum' aggregation method.
    
    Returns:
        np.ndarray: Array of frame embeddings [k, emb_dim].
    """

    embs = []
    for frame in frames_np:
        e = emb_model.predict(np.expand_dims(frame, axis=0), verbose=0) # input dim: [1, H, W, 3]
        embs.append(e)
    
    return np.vstack(embs) # output dim: [k, emb_dim]


def aggregate_video_embeddings(embs: list[np.ndarray], agg_method: str = 'mean') -> np.ndarray:
    """
    Aggregates frame embeddings into a single video embedding.
    Args:
        embs (list[np.ndarray]): List of frame embeddings.
        agg_method (str): 'mean' or 'sum' aggregation method.
    
    Returns:
        np.ndarray: Aggregated video embedding.    
    """

    if agg_method == 'mean':
        return np.mean(np.vstack(embs), axis=0)
    elif agg_method == 'sum':
        return np.sum(np.vstack(embs), axis=0)



def main():
    # random states
    np.random.seed(42)
    tf.random.set_seed(42)

    EMISSIONS_OUTPUT_DIR = str(cfg.EMISSIONS_OUTPUT_DIR)
    EMBEDDING_DIR = str(cfg.EMBEDDING_DIR)
    VIDEO_ROOT = str(cfg.SAMPLED_OUTPUT_DIR)
    IMG_SIZE = cfg.SIZE_FOR_XCEPTION

    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR, exist_ok=True)
    if not os.path.exists(EMISSIONS_OUTPUT_DIR):
        os.makedirs(EMISSIONS_OUTPUT_DIR, exist_ok=True)

    # CodeCarbon tracker
    tracker = EmissionsTracker(output_dir = EMISSIONS_OUTPUT_DIR, project_name='deepfake_recognition_embedding_creation')
    tracker.start()

    if cfg.USE_CACHED_EMBEDDINGS:
        if all(os.path.exists(os.path.join(EMBEDDING_DIR, f'train_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')) for split in ['train', 'val', 'test']):
            print('Video embeddings already exist and USE_CACHED_EMBEDDINGS is True. Exiting...')
        else: 
            print('USE_CACHED_EMBEDDINGS is True, but the aggregation method changed.')
            print(f'Recomputing aggregated embeddings with {cfg.EMBEDDING_AGGREGATION}...')

            for split in ['train', 'val', 'test']:
                original_csv = os.path.join(EMBEDDING_DIR, f'{split}_all_video_embeddings.csv')

                big_df = pd.read_csv(original_csv)
                columns = [f'e{j}' for j in range(2048)] 
                agg_df = pd.DataFrame(columns = columns+ ['label', 'vid_name'])

                for i in range(0, len(big_df), cfg.FRAMES_PER_VIDEO):
                    chunk = big_df.iloc[i:i+cfg.FRAMES_PER_VIDEO] # each chunk is a DataFrame of FRAMES_PER_VIDEO rows
                    vid_embs = chunk[columns].to_numpy()

                    agg_emb = aggregate_video_embeddings(vid_embs, cfg.EMBEDDING_AGGREGATION).flatten()

                    # populate agg_df (one column for each agg_emb component)
                    agg_row_df = pd.DataFrame([agg_emb], columns=columns)
                    agg_row_df['label'] = chunk['label'].iloc[0]
                    agg_row_df['vid_name'] = chunk['vid_name'].iloc[0]
                    agg_df = pd.concat([agg_df, agg_row_df], ignore_index=True)
                
                agg_emb_csv = os.path.join(EMBEDDING_DIR, f'{split}_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')
                agg_df.to_csv(agg_emb_csv, index=False)
                print(f'Saved aggregated embeddings successfully to: {agg_emb_csv}!')
    else:
        print('Computing video embeddings from scratch...')
        
        # Kera's Xception model for embeddings
        base = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        emb_model = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))  
        emb_dim = emb_model.output_shape[-1] # output dim: 2048
        print(f'Xception embedding model loaded. Embedding size: {emb_dim}')

        for split in ['train', 'val', 'test']:
            columns = [f'e{j}' for j in range(emb_dim)]
            big_df = pd.DataFrame(columns = columns + ['label', 'vid_name'])
            agg_df = pd.DataFrame(columns = columns + ['label', 'vid_name'])

            for label in ['real', 'fake']:
                folder = os.path.join(VIDEO_ROOT, split, label)
                videos = [f'{VIDEO_ROOT}/{split}/{label}/{f}' for f in os.listdir(folder) if f.endswith('.mp4')]
                
                print(f'{split}/{label}: {len(videos)} videos')
                for vp in tqdm(videos, desc=f'{split}/{label}'):
                    vid_name = os.path.splitext(os.path.basename(vp))[0] # without the .mp4 extension

                    print(f'\n Processing video: {vid_name}.mp4...')
                    frames = extract_video_frames_uniform(vp, IMG_SIZE, k = cfg.FRAMES_PER_VIDEO)
                    x = preprocess_for_Xception(frames, IMG_SIZE)

                    vid_embs = build_frame_embeddings(emb_model, x) 
                    agg_emb = aggregate_video_embeddings(vid_embs, cfg.EMBEDDING_AGGREGATION).flatten()

                    # populate big_df (all embeddings) and agg_df (one column for each agg_emb component)
                    vid_df = pd.DataFrame(vid_embs, columns=columns)
                    vid_df['label'] = label
                    vid_df['vid_name'] = vid_name
                    big_df = pd.concat([big_df, vid_df], ignore_index=True)

                    agg_row_df = pd.DataFrame([agg_emb], columns=columns)
                    agg_row_df['label'] = label
                    agg_row_df['vid_name'] = vid_name
                    agg_df = pd.concat([agg_df, agg_row_df], ignore_index=True)

            all_embs_csv = os.path.join(EMBEDDING_DIR, f'{split}_all_video_embeddings.csv')
            big_df.to_csv(all_embs_csv, index=False)
            print(f'Saved all embeddings successfully to: {all_embs_csv}!')

            agg_emb_csv = os.path.join(EMBEDDING_DIR, f'{split}_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')
            agg_df.to_csv(agg_emb_csv, index=False)
            print(f'Saved aggregated embeddings successfully to: {agg_emb_csv}!')
    
    emissions = tracker.stop()
    print(f"\n Emissions tracked: {emissions} kg CO2")


if __name__ == "__main__":
    main()

