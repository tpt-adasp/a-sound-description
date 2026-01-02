from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import os
import torch.nn as nn
import torch



class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class FSD50K(AudioDataset):
    base_folder = 'FSD50K'
    audio_dir = 'audio'
    label_col = 'label'
    file_col = 'file_path'
    meta = {
        'filename': os.path.join('metadata.csv'),
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations

        print("Loading audio files")
        # self.df['filename'] = os.path.join(self.root, self.base_folder, self.audio_dir) + os.sep + self.df['filename']
        self.df['category'] = self.df['label'].str.replace('_',' ')

        # Drop repeated file paths
        df_file_paths = self.df.drop_duplicates(subset=['file_path'])
        print("Number of unique audio files: ", len(df_file_paths))
        print(df_file_paths.head())

        for _, row in tqdm(df_file_paths.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, row[self.file_col])
            # self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)
        
        # Print number of audio files
        print("Number of audio files: ", len(self.audio_paths)) # 102394

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        self.df = pd.read_csv(path)

        self.class_to_idx = {}
        self.classes = self.df[self.label_col].unique()
        self.classes = [x.replace('_',' ') for x in self.classes]

        self.classes.sort()

        for i, category in enumerate(self.classes):
            self.class_to_idx[category.replace(' ','_')] = i
        # Print classes along with their indices
        print("Classes: ", self.classes)
        # Count the number of classes
        print("Number of classes: ", len(self.classes)) # 200

    def compose_one_hot_target(self, file_path):
        # print("File path: ", file_path)
        
        # Remove root and base folder from file_path
        file_path = file_path.split(self.root+"/FSD50K/")[1]

        metadata_file = self.df[self.df.file_path == file_path]
        labels_names = list(metadata_file['label'])
        labels_idx = [self.class_to_idx[label] for label in labels_names]

        one_hot_target = torch.zeros(200)
        one_hot_target[labels_idx] = 1
        one_hot_target = one_hot_target.reshape(1, -1)

        return one_hot_target, labels_names


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file_path = self.audio_paths[index]
        one_hot_target, _ = self.compose_one_hot_target(file_path)
        target = []

        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        pass


