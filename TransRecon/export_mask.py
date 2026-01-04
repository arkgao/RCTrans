import argparse
import logging
import os
import random
from shutil import copyfile

import cv2

import numpy as np
import torch
from pyhocon import ConfigFactory
from tqdm import tqdm
from glob import glob

from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.RayTracer import RayTracer
from models.utils import Logger


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def cal_MAE(gt, pred):
    diff = gt - pred
    return np.mean(np.abs(diff))

def find_largest_region(mask):
    mask = mask.astype(np.uint8)
    countours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(countours)):
        area.append(cv2.contourArea(countours[i]))
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(countours[max_idx])
    for j in range(len(countours)):
        if j != max_idx:
            cv2.fillPoly(mask, [countours[j]],0)
    return mask

# use open operation to clean the noise in mask
def clean_mask(mask):
    kernel = np.ones([5,5], dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask
    

def get_margin(mask, size=5):
    kernel = np.ones([size,size], dtype=np.uint8)
    dilate = cv2.dilate(mask, kernel,iterations=1) 
    erosion = cv2.erode(mask, kernel)
    margin = dilate - erosion
    return margin

class Exper:
    """
        Load the trained model in stage1 and estimate the object mask
    """
    def __init__(self, conf_path, case='CASE_NAME', exp_name = 'export_mask', val_error = False, real_data=False):
        self.exp_name = exp_name
        self.device = torch.device('cuda')
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
            
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])

        self.use_white_bkgd = False

        self.model_list = []

        # Networks
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        
        # Here the tracer would load a default bounding box (unit cube), and the bounding box would be used to cut the weight
        self.tracer = RayTracer( self.sdf_network,
                                    self.deviation_network,
                                    self.dataset.bounding_box,
                                    sample_full=True,
                                    **self.conf['model.tracer'])

        
        # Load min_z if exists
        if os.path.exists(os.path.join(self.conf['dataset.data_dir'],'min_z.txt')):
            self.min_z = float(np.loadtxt(os.path.join(self.conf['dataset.data_dir'],'min_z.txt')))
        else:
            self.min_z = None

        self.val_error = val_error
        if self.val_error:
            mask_dir = os.path.join(self.conf['dataset.data_dir'], 'mask')
            self.dataset.load_mask(mask_dir)
        
        self.real_data = real_data

        self.reload_trained_model()

        self.file_backup()
    
    def file_backup(self):
        dir_lis = self.conf['general.recording']
        basedir = os.path.join(self.base_exp_dir, self.exp_name, 'recording')
        os.makedirs(basedir, exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(basedir, dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py' and (not f_name.startswith('tmp')):
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(basedir, 'config.conf'))

    def find_latest_ckpt(self,path):
        model_list = sorted(glob(os.path.join(path,'*.pth')))
        return model_list[-1]

    def reload_trained_model(self):
        stage1_model_path = self.find_latest_ckpt(os.path.join(self.base_exp_dir,'stage1','checkpoints'))
        logging.info('Load trained stage1 model from: {}'.format(stage1_model_path))
        checkpoint = torch.load(stage1_model_path, map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])

    def circle_mask(self,rays_o, rays_d, r=1.0):
        t = -rays_o[:,2]/rays_d[:,2]
        end_point = rays_o + t*rays_d
        circle_mask = torch.sqrt(end_point[:,0]**2+end_point[:,1]**2)<0.95
        return circle_mask
       
    def export_mask(self):
        basedir = os.path.join(self.base_exp_dir, self.exp_name)
        os.makedirs(basedir,exist_ok=True)
        
        batch_size = self.conf['export_mask.batch_size']
            
        img_num = len(self.dataset.images_lis)
        img_idx_list = np.arange(0,img_num)
        
        error_list = []
        error_dir = os.path.join(basedir,'error')
        mask_dir = os.path.join(basedir,'mask')
        margin_dir = os.path.join(basedir, 'margin')
        
        if self.real_data:
            vis_dir = os.path.join(basedir,'vis')
            os.makedirs(vis_dir, exist_ok=True)
            fullmask_dir = os.path.join(basedir,'fullmask')
            os.makedirs(fullmask_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(margin_dir, exist_ok=True)
        
        logger = Logger(error_dir)
        
        for img_idx in img_idx_list:
            rays_o, rays_d = exper.dataset.gen_rays_at(img_idx, resolution_level=1)
            H, W, _ = rays_o.shape
            
            rays_o = rays_o.reshape(-1, 3).split(batch_size)
            rays_d = rays_d.reshape(-1, 3).split(batch_size)

            mask_list = []
            fullmask_list = []
            
            for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
                render_output = self.tracer.ray_tracing(rays_o_batch,   
                                                     rays_d_batch
                                                    )
                
                sphere_mask = render_output['sphere_mask']
                mask = render_output['weights_sum'].squeeze(1) > 0.1
                mask = mask & sphere_mask
                
                if self.min_z is not None:
                    end_point = rays_o_batch + render_output['depth']*rays_d_batch
                    extra_mask = end_point[:,2]>self.min_z
                    fullmask = mask
                    mask = mask & extra_mask
                mask_list.append(mask.detach().cpu().numpy())
                fullmask_list.append(fullmask.detach().cpu().numpy())
                    
            pred = np.concatenate(mask_list, axis=0).reshape(H,W,-1).astype(np.float32)*255
            pred = find_largest_region(pred)
            
            # For real data, we extra output the mask before min_z filtering, to assist users in verifying whether min_z is correct
            if self.real_data:
                fullmask = np.concatenate(fullmask_list, axis=0).reshape(H,W,-1).astype(np.float32)*255
                fullmask = find_largest_region(fullmask)
                cv2.imwrite(os.path.join(fullmask_dir,'{}.png'.format(str(img_idx).zfill(3))),fullmask)
            
            # for real data, there is a little noise caused by the support
            # we use a open operation to clean the noise
            if self.real_data:
                pred = clean_mask(pred)
            
            cv2.imwrite(os.path.join(mask_dir,'{}.png'.format(str(img_idx).zfill(3))),pred)
            margin = get_margin(pred)
            cv2.imwrite(os.path.join(margin_dir,'{}.png'.format(str(img_idx).zfill(3))),margin)
            
            if len(pred.shape) == 2:
                pred = pred[:,:,np.newaxis]
            
            if self.val_error:
                gt = self.dataset.mask_at(img_idx, resolution_level=1)
                # if len(gt.shape) == 3:
                #     gt = gt[:,:,0]
                if len(gt.shape) == 2:
                    gt = gt[:,:,np.newaxis]
                pred = pred / 255.0
                error = cal_MAE(gt,pred)
                show_img = np.vstack([np.hstack([pred,gt]),np.hstack([pred-gt,gt-pred])])
                cv2.putText(show_img, 'MAE: {:.4f}'.format(error), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1)
                h,w = show_img.shape[:2]
                cv2.putText(show_img, 'pred', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1)
                cv2.putText(show_img, 'gt', (10+w//2,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1)
                cv2.putText(show_img, 'pred-gt', (10,100+h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1)
                cv2.putText(show_img, 'gt-pred', (10+w//2,100+h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1)
                cv2.imwrite(os.path.join(basedir,'error','{}.png'.format(str(img_idx).zfill(3))),show_img*255)
                error_list.append(error)
                logger.printandwrite('img {}:  mask error {}'.format(img_idx, error))
            elif self.real_data:
                # If there is no ground-truth, we visualize the result by blending the mask and input image
                img = self.dataset.image_at(img_idx)
                color_mask = np.zeros_like(img)
                color_mask[:,:,0:1] = pred
                alpha = 0.5
                vis = img * (1-alpha) + color_mask * alpha
                vis = np.concatenate([img,vis],axis=1)
                cv2.imwrite(os.path.join(vis_dir,'{}.png'.format(str(img_idx).zfill(3))),vis*255)

        if self.val_error:
            logger.printandwrite('mean error: {}'.format(np.mean(np.array(error_list))))


if __name__ == '__main__':
    print('Hello Ark')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/export_mask.conf')    
    parser.add_argument('--exp_name', type=str, default='export_mask')       
    parser.add_argument('--gpu', type=int, default=0)                    
    parser.add_argument('--case', type=str, default='cat')       
    parser.add_argument('--val_error', action='store_true')
    parser.add_argument('--real_data', action='store_true')
    
    args = parser.parse_args()

    print("Deal case {}".format(args.case))
    
    torch.cuda.set_device(args.gpu)
    exper = Exper(args.conf, args.case, args.exp_name, args.val_error, args.real_data)
    exper.export_mask()
    
    


