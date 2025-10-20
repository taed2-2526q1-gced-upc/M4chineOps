from pathlib import Path

# from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
# load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f'PROJ_ROOT path is: {PROJ_ROOT}')

# data directories
GLOBAL_DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = GLOBAL_DATA_DIR / 'raw'

REAL_DATA_SUBDIR = RAW_DATA_DIR / 'original_sequences/youtube/c40/videos'
FAKE_DATA_SUBDIR = RAW_DATA_DIR / 'manipulated_sequences/deepfakes/c40/videos'

SAMPLED_OUTPUT_DIR = GLOBAL_DATA_DIR / 'sampled_videos'
METADATA_DIR = GLOBAL_DATA_DIR / 'metadata'

EMBEDDING_DIR = GLOBAL_DATA_DIR / 'embeddings'
USE_CACHED_EMBEDDINGS = False  # if True, use previously extracted embeddings if they exist

# parameters for sampling and modelling
N_SAMPLES_PER_CLASS = 250

SIZE_FOR_XCEPTION = (299, 299)  # (height, width)
FRAMES_PER_VIDEO = 10
EMBEDDING_AGGREGATION = 'mean'  # 'mean' or 'sum'

MODELS_DIR = PROJ_ROOT / 'models'


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
except ModuleNotFoundError:
    pass
