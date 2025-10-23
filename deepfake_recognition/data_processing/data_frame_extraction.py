import pandas as pd
import os
import cv2
import time
from tqdm import tqdm
from mtcnn import MTCNN

import deepfake_recognition.config as cfg

#################################################
#         Face Frame Extraction with Boxes      #
# --------------------------------------------- #
# - Extracts faces using MTCNN                  #
# - Saves cropped faces and debug frames        #
# - Also saves frames with bounding boxes       #
#################################################

def extract_face_frames_from_video(detector: MTCNN, video_info: dict, split: str, num_frames_to_extract: int):
    """
    Extracts faces from a video and saves:
      - Cropped faces
      - Original frames (debug)
      - Frames with bounding boxes (visual debug)
    """

    label = video_info.get('label', 0)
    filepath = video_info.get('filepath')

    # Si el video viene desde la API (ruta temporal)
    if not os.path.exists(filepath):
        path_parts = filepath.split('/')
        if label == 0:
            filepath = os.path.join(cfg.RAW_DATA_DIR, 'original_sequences', path_parts[0], 'c40/videos', path_parts[1])
        elif label == 1:
            filepath = os.path.join(cfg.RAW_DATA_DIR, 'manipulated_sequences', path_parts[0], 'c40/videos', path_parts[1])

    if not os.path.exists(filepath):
        print(f"Error: No se encontró el archivo de video en {filepath}")
        return []

    filename = os.path.splitext(os.path.basename(filepath))[0]
    cap = cv2.VideoCapture(filepath)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {filepath}")
        return []

    frames = int(video_info.get('frames', cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    step = max(1, frames // num_frames_to_extract)

    print(f"📹 Procesando video: {filename}")
    print(f"🔹 Frames totales: {frames}, paso de muestreo: {step}")

    frame_paths = []
    for i in tqdm(range(num_frames_to_extract), desc=f"Extracting Faces ({filename})"):
        frame_index = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f" No se pudo leer el frame {i} en {filepath}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame_rgb)

        # Guardar el frame original (debug)
        debug_frame_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'{split}_data', f'{filename}_DEBUG_FRAME_{i}.jpg')
        cv2.imwrite(debug_frame_path, frame)

        # Si hay detección de cara
        if results:
            x, y, width, height = results[0]['box']
            x, y = max(0, x), max(0, y)

            #Dibuja un recuadro sobre el frame original
            boxed_frame = frame.copy()
            cv2.rectangle(boxed_frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            boxed_frame_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'{split}_data', f'{filename}_BOXED_FRAME_{i}.jpg')
            cv2.imwrite(boxed_frame_path, boxed_frame)

            # Recorta y guarda el rostro
            face = frame[y:y + height, x:x + width]
            face_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'{split}_data', f'{filename}_FRAME_{i}.jpg')
            cv2.imwrite(face_path, face)
            frame_paths.append(face_path)

            print(f"Cara detectada y guardada en: {face_path}")
            print(f"Frame con recuadro guardado en: {boxed_frame_path}")

        else:
            print(f"No se detectó rostro en frame {i}, guardado como debug: {debug_frame_path}")

    cap.release()
    print(f"Total de caras detectadas en {filename}: {len(frame_paths)}\n")
    return frame_paths


def main():
    detector = MTCNN()
    num_frames_to_extract = 10

    for split in ['train', 'val', 'test']:
        print(f"🚀 Procesando split: {split}")
        csv_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'{split}_data.csv')

        if not os.path.exists(csv_path):
            print(f"No se encontró el archivo CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        output_dir = os.path.join(cfg.PROCESSED_DATA_DIR, f'{split}_data')
        os.makedirs(output_dir, exist_ok=True)

        start = time.time()
        df['frame_paths'] = df.apply(
            lambda row: extract_face_frames_from_video(detector, row, split, num_frames_to_extract), axis=1)
        end = time.time()

        print(f"Video face frames guardados en {output_dir} en {end - start:.2f}s")
        df.to_csv(csv_path, index=False)
        print(f"CSV actualizado: {csv_path}\n")


if __name__ == '__main__':
    main()
