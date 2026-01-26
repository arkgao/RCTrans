import glob
import os.path as osp
from torch.utils.data import Dataset
from dataset.dataset_utils import read_list, read_img, read_mask, read_corres
from utils import get_root_logger
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CorresDataset(Dataset):
    """Dataset for correspondence training"""
    def __init__(self,opt):
        split = opt['split']
        super().__init__()
        self.opt = opt
        self.split = split

        if split == 'train':
            self.list_file = osp.join(opt['dataroot'], opt['list_file'])
            self.dir = osp.join(opt['dataroot'], 'train')
        elif split == 'val':
            self.list_file = osp.join(opt['dataroot'], opt['list_file'])
            self.dir = osp.join(opt['dataroot'], 'val')
        
        logger = get_root_logger()
        logger.info('Loading data from {}'.format(self.dir))
        self.basename_list = read_list(self.list_file)


    def __len__(self):
        return len(self.basename_list)

    def __getitem__(self,index):
        basename = self.basename_list[index]
        input_path = osp.join(self.dir, basename + '_image.png')
        bg_path = osp.join(self.dir, basename + '_background.png')
        corres_path = osp.join(self.dir, basename + '_correspondence.flo')
        validmask_path = osp.join(self.dir, basename + '_valid_mask.png')

        backgound  = read_img(bg_path)
        input_img  = read_img(input_path)
        corres     = read_corres(corres_path)
        valid_mask = read_mask(validmask_path)

        sample = {
            'input_img'  : input_img,
            'background' : backgound,
            'correspondence': corres,
            'valid_mask' : valid_mask,
            }

        return sample

@DATASET_REGISTRY.register()
class ReconDataset(Dataset):
    """Dataset for reconstruction inference"""
    def __init__(self,opt):
        input_folder = opt['input_folder']
        self.split = opt['split']
        super().__init__()
        logger = get_root_logger()
        logger.info('Loading data from {}'.format(input_folder))
        self.input_img_list = glob.glob(osp.join(input_folder,'input_*.png'))
        self.mask_list = glob.glob(osp.join(input_folder,'mask_*.png'))
        self.bk_img_list = glob.glob(osp.join(input_folder,'background_*.png'))
        self.input_img_list.sort()
        self.bk_img_list.sort()
        self.mask_list.sort()

    def __len__(self):
        return len(self.input_img_list)

    def __getitem__(self,index):
        input_img = read_img(self.input_img_list[index])
        background = read_img(self.bk_img_list[index])
        mask = read_mask(self.mask_list[index])

        sample = {
                'input_img'  : input_img,
                'valid_mask'   : mask,
                'background' : background,
            }
        return sample
