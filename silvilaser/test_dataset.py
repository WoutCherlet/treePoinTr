import os
from utils import parser
from tools import builder
from datasets import build_dataset_from_cfg
from utils.config import *
import torch
from utils.misc import *



def main():

    cfg_path = "cfgs/dataset_configs/CLS_test.yaml"

    config = cfg_from_yaml_file(cfg_path)
    config.subset = "test"

    dataset = build_dataset_from_cfg(config)

    # print(dataset.file_list[:10])
    sampler = None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 2,
                                            num_workers = 1,
                                            drop_last = False,
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):

        print(taxonomy_ids)
        print(model_ids)
        print(data[0].shape)
        print(data[1].shape)
        if idx>= 5:
            break



if __name__ == "__main__":
    main()
