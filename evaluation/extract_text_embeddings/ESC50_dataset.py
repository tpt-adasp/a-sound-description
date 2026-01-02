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


class ESC50(AudioDataset):
    base_folder = 'ESC-50'
    filename = "ESC-50-master.zip"
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'label'
    file_col = 'file_id'
    meta = {
        'filename': os.path.join('metadata.csv'),
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        self.df['category'] = self.df['label'].str.replace('_',' ')

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_',' ') for x in sorted(self.df[self.label_col].unique())]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category.replace(' ','_')] = i
        # Print classes along with their indices
        print("Classes: ", self.classes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1,-1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        pass



