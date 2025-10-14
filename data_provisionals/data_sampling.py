import os, random, shutil
import os

output_dir = "data/ffpp_sampled"
os.makedirs(output_dir, exist_ok=True)

def sample_split(src_real, src_fake, dst_base, n_per_class=200):
    """Sample real and fake videos and split into train/val/test."""
    splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}

    # List all mp4 files
    real_videos = [f for f in os.listdir(src_real) if f.endswith(".mp4")]
    fake_videos = [f for f in os.listdir(src_fake) if f.endswith(".mp4")]

    # Sample
    real_sample = random.sample(real_videos, min(n_per_class, len(real_videos)))
    fake_sample = random.sample(fake_videos, min(n_per_class, len(fake_videos)))

    def make_split(videos, src_dir, label):
        random.shuffle(videos)
        n = len(videos)
        start = 0
        for split, ratio in splits.items():
            end = start + int(ratio * n)
            split_videos = videos[start:end]
            out_dir = os.path.join(dst_base, split, label)
            os.makedirs(out_dir, exist_ok=True)
            for v in split_videos:
                shutil.copy(os.path.join(src_dir, v), os.path.join(out_dir, v))
            print(f"{label} - {split}: {len(split_videos)} videos")
            start = end

    make_split(real_sample, src_real, "real")
    make_split(fake_sample, src_fake, "fake")

def main():
    src_real = "data/raw/original_sequences/youtube/c40/videos"
    src_fake = "data/raw/manipulated_sequences/Deepfakes/c40/videos"
    dst_base = "data/ffpp"
    n_per_class = 800
    sample_split(src_real, src_fake, dst_base, n_per_class)

if __name__ == "__main__":
    main()
