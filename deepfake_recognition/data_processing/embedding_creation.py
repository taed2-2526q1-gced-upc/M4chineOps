import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

import deepfake_recognition.config as cfg


def extract_video_frames_uniform(path: str, IMG_SIZE: tuple[int, int], k: int = 10) -> list[np.ndarray]:
    """
    Extract frames at uniform intervals from a video file.
    Save the frames as JPEG files in VIDEO_FRAME_DIR/split/video_name/
    
    Args:
        path (str): Path to the video file.
        k (int): Number of frames to extract from each video.
    """

    cap = cv2.VideoCapture(path)
        
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
    """

    if agg_method == 'mean':
        return np.mean(np.vstack(embs), axis=0)
    elif agg_method == 'sum':
        return np.sum(np.vstack(embs), axis=0)



def main():
    # random states
    np.random.seed(42)
    tf.random.set_seed(42)

    EMBEDDING_DIR = str(cfg.EMBEDDING_DIR)
    VIDEO_ROOT = str(cfg.SAMPLED_OUTPUT_DIR)
    IMG_SIZE = cfg.SIZE_FOR_XCEPTION

    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR, exist_ok=True)

    if cfg.USE_CACHED_EMBEDDINGS:
        if all(os.path.exists(os.path.join(EMBEDDING_DIR, f'train_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')) for split in ['train', 'val', 'test']):
            print('Video embeddings already exist and USE_CACHED_EMBEDDINGS is True. Exiting...')
        else: 
            print('USE_CACHED_EMBEDDINGS is True, but the aggregation method changed.')
            print(f'Recomputing aggregated embeddings with {cfg.EMBEDDING_AGGREGATION}...')

            for split in ['train', 'val', 'test']:
                original_csv = os.path.join(EMBEDDING_DIR, f'{split}_all_video_embeddings.csv')

                agg_df_columns = [f'e{j}' for j in range(2048)] + ['label']
                agg_df = pd.DataFrame(columns = agg_df_columns)

                for vid in pd.read_csv(original_csv).itertuples():
                    vid_embs = [getattr(vid, f'frame_emb_{i+1}') for i in range(cfg.FRAMES_PER_VIDEO)]
                    agg_emb = aggregate_video_embeddings(vid_embs, cfg.EMBEDDING_AGGREGATION).flatten()

                    # populate agg_df (one column for each agg_emb component)
                    agg_row_dict = {f'e{i}': agg_emb[i] for i in range(len(agg_emb))}
                    agg_row_dict['label'] = vid.label
                    agg_df.loc[vid.Index] = agg_row_dict
                
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
            df_columns = [f'frame_emb_{i+1}' for i in range(cfg.FRAMES_PER_VIDEO)] + [f'agg_emb', 'label'] 
            big_df = pd.DataFrame(columns = df_columns)

            agg_df_columns = [f'e{j}' for j in range(emb_dim)] + ['label']
            agg_df = pd.DataFrame(columns = agg_df_columns)

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
                    row_dict = {f'frame_emb_{i+1}': vid_embs[i] for i in range(len(vid_embs))} # frame embeddings
                    row_dict['label'] = label
                    row_dict['agg_emb'] = agg_emb # aggregated embedding
                    big_df.loc[vid_name] = row_dict

                    agg_row_dict = {f'e{i}': agg_emb[i] for i in range(emb_dim)}
                    agg_row_dict['label'] = label
                    agg_df.loc[vid_name] = agg_row_dict

            all_embs_csv = os.path.join(EMBEDDING_DIR, f'{split}_all_video_embeddings.csv')
            big_df.to_csv(all_embs_csv, index=False)
            print(f'Saved all embeddings successfully to: {all_embs_csv}!')

            agg_emb_csv = os.path.join(EMBEDDING_DIR, f'{split}_{cfg.EMBEDDING_AGGREGATION}_video_embeddings.csv')
            agg_df.to_csv(agg_emb_csv, index=False)
            print(f'Saved aggregated embeddings successfully to: {agg_emb_csv}!')


if __name__ == "__main__":
    main()

