import os
import pandas as pd
import yaml 

from sklearn.model_selection import train_test_split

import deepfake_recognition.config as cfg

#################################################
#                 Data Splitting                #
# --------------------------------------------- #
# - creating train, validation and test splits  #
#################################################


def split_data(csv_path: str, output_dir: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        csv_path (str): Path to the CSV file containing video metadata.
        output_dir (str): Directory to save the split CSV files.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Random seed for reproducibility.
    """

    # create a single DataFrame from all metadata CSV files
    csv_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    train_val_df, test_df = train_test_split(df, test_size = test_size, random_state = random_state, stratify = df['label']) # stratify to maintain label distribution
    
    val_relative_size = val_size / (1 - test_size)  
    train_df, val_df = train_test_split(train_val_df, test_size = val_relative_size, random_state = random_state, stratify = train_val_df['label'])

    for split_name, split_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        output_path = os.path.join(output_dir, f'{split_name}_data.csv')
        split_df.to_csv(output_path, index=False)
        print(f'Sucessfully saved {split_name} split in {split_name}_data.csv with {len(split_df)} entries!')
        print(f'Label distribution: {split_df["label"].value_counts()}')


def main():

    csv_path = str(cfg.METADATA_DIR)
    output_dir = str(cfg.PROCESSED_DATA_DIR)

    os.makedirs(output_dir, exist_ok=True)
    split_data(csv_path, output_dir)


if __name__ == '__main__':
    main()