##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--data_cfg', type=str, required=True, help='data config file')   
    parser.add_argument("--data_root", "-d", type=str, required=True, help="Data root where blocks are stored")
    parser.add_argument(
        "--out_dir", "-o", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.model_config is not None
    assert args.model_checkpoint is not None

    return args

def inference_block(model, block_path, args):

    # read block
    block = np.load(os.path.join(args.data_root, block_path))
    partial_pc = block['partial']
    
    transform = Compose([{
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    partial_tensor = transform({'input': partial_pc})
    # inference
    ret = model(partial_tensor['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    target_path = os.path.join(args.out_dir, os.path.splitext(block_path)[0])
    os.makedirs(target_path, exist_ok=True)

    np.save(os.path.join(target_path, 'compl_fine.npy'), dense_points)
    
    if args.save_vis_img:
        input_img = misc.get_ptcloud_img(partial_tensor['input'].numpy())
        dense_img = misc.get_ptcloud_img(dense_points)
        cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
        cv2.imwrite(os.path.join(target_path, 'compl_fine.jpg'), dense_img)
    
    return

def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)

    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    data_cfg_folder = config['dataset']["test"]['_base_']['DATA_FOLDER']
    data_cfg_file = os.path.join(data_cfg_folder, "test.txt")
    with open(data_cfg_file, 'r') as f:
        for block_file in f.readlines():
            inference_block(base_model, block_file, args)
    
if __name__ == '__main__':
    main()