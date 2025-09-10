import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
import data_transforms

@DATASETS.register_module()
class CLS(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_FOLDER
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        self.file_list = self._get_file_list()
        
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def _get_transforms(self, subset):
            if subset == 'train':
                return data_transforms.Compose([{
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                },{
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])
            else:
                return data_transforms.Compose([ {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])

    def _get_file_list(self):
        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = "CLS"
            model_id, path = line.split("_", 1)
            file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': path
            })
        return file_list

    def __getitem__(self, idx):

        sample = self.file_list[idx]

        data_npz = np.load(os.path.join(self.pc_path, sample["model_id"], "blocks", sample['file_path']))

        data_partial = torch.from_numpy(data_npz["partial"]).float()
        data_gt = torch.from_numpy(data_npz["complete"]).float()

        return sample['taxonomy_id'], sample['model_id'], (data_partial, data_gt)

    def __len__(self):
        return len(self.file_list)