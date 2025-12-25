import cv2 
import os
import logging
import argparse
import numpy as np
import random
from glob import glob
import torch
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork,SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import BackgroundRenderer
from models.utils import Logger
from matplotlib import cm

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def colormap(diff,thres):
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = cm.jet(diff_norm)[:,:, :3]
    return diff_cm[:,:,::-1]

class Exper:
    def __init__(self, conf, exp_name='render_envmap', x_fov=None):
        self.device = torch.device('cuda')
        self.exp_name = exp_name
        
        # Configuration
        self.conf = conf
        
        # For synthetic data, we use the same camera fov as its input images
        # For real data, we manually set the fov in config file
        self.fov = x_fov
            
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])

        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.reload_trained_model()
        
        self.renderer = BackgroundRenderer(nerf_outside=self.nerf_outside,
                                    envmap=None)

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

        copyfile(self.conf['conf_path'], os.path.join(basedir, 'config.conf'))

    def find_latest_ckpt(self,path):
        model_list = sorted(glob(os.path.join(path,'*.pth')))
        return model_list[-1]

    def reload_trained_model(self):
        # Load checkpoint of stage1 for scene network
        load_folder = self.conf.get_string('load_folder',default='stage1')
        model_path = self.find_latest_ckpt(os.path.join(self.base_exp_dir, load_folder,'checkpoints'))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        logging.info('Load trained nerf++ from: {}'.format(model_path))

    def render_view_background_blend(self):
        """
            For each input view in dataset, render a background image without the object, and save
            Blend the rendered background with the input image using the mask.
        """
        self.dataset.load_mask(os.path.join(self.base_exp_dir,'export_mask','mask'))
        exp_dir = os.path.join(self.base_exp_dir, self.exp_name, 'view')
        batch_size = 256
        os.makedirs(exp_dir, exist_ok=True)
        print("Rendering background for each view...")
        for idx in tqdm(range(self.dataset.n_images)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
            H, W, _ = rays_o.shape        
            input_img = self.dataset.image_at(idx)
            mask = self.dataset.mask_at(idx)
            mask = (255*mask).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations=1)
            mask = mask / 255
            mask_flat = mask.reshape(H * W)
            rays_o = rays_o.reshape(-1, 3)[mask_flat == 1].split(batch_size)
            rays_d = rays_d.reshape(-1, 3)[mask_flat == 1].split(batch_size)
            color_list = []
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                render_output = self.renderer.render(rays_o_batch, rays_d_batch)
                color_list.append(render_output['color_fine'].detach().cpu().numpy())

            color = np.concatenate(color_list, axis=0).reshape(-1, 3)
            img = input_img * (1 - mask.reshape(H,W,1))
            img[mask == 1] = color
            cv2.imwrite(os.path.join(exp_dir,f'background_{idx}.png'),img*255)

    def render_view_background(self):
        """
            For each input view in dataset, render a background image without the object, and save
            Render the whole backgruond images using nerf++
        """
        exp_dir = os.path.join(self.base_exp_dir, self.exp_name, 'view')
        batch_size = 256
        os.makedirs(exp_dir, exist_ok=True)
        self.dataset.set_xfov(self.fov)
        print("Rendering background for each view...")
        for idx in tqdm(range(self.dataset.n_images)):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
            rays_o = torch.zeros_like(rays_d).to(self.device)+rays_d
            H, W, _ = rays_o.shape        
            rays_o = rays_o.reshape(-1, 3).split(batch_size)
            rays_d = rays_d.reshape(-1, 3).split(batch_size)
            color_list = []
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                render_output = self.renderer.render(rays_o_batch, rays_d_batch)
                color_list.append(render_output['color_fine'].detach().cpu().numpy())

            color = np.concatenate(color_list, axis=0).reshape(-1, 3)
            color = color.reshape(H, W, 3).astype(np.float32)
            cv2.imwrite(os.path.join(exp_dir,f'background_{idx}.png'),color*255)
        
    def render_envmap(self,resolution=(512,1024)):
        """
            Render the environment map using nerf++
        """
        print("Rendering envmap...")
        exp_dir = os.path.join(self.base_exp_dir, self.exp_name)

        batch_size = 256
        r = 3
        phi_sample, theta_sample = torch.meshgrid(torch.linspace(0, torch.pi, resolution[0]),
                                                  torch.linspace(0, 2*torch.pi, resolution[1]), indexing='ij')
        
        location = torch.stack([r*torch.sin(phi_sample)*torch.sin(theta_sample),
                                r*torch.sin(phi_sample)*torch.cos(theta_sample),
                                r*torch.cos(phi_sample)],dim=-1)

        rays_o = location.to('cuda')
        rays_d = rays_o / r
        
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)

        color_list = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            render_output = self.renderer.render(rays_o_batch,
                                                rays_d_batch)
            color_list.append(render_output['color_fine'].detach().cpu().numpy())
        color = np.concatenate(color_list,axis=0).reshape(resolution[0], resolution[1],3).astype(np.float32)
        
        # We found there are some black points in the envmap. It's may because the rays in input view are relatively sparse and the nerf++ is not fully trained. So we use inpaint to fill the black points.
        color = (255*color).clip(0,255).astype(np.uint8)
        cv2.imwrite(os.path.join(exp_dir,'raw_envmap.png'),color)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,2] < 50)
        mask = 255*mask.astype(np.uint8)
        mask = mask - cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
        color = cv2.inpaint(color, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(exp_dir,'envmap.png'),color)

        

if __name__ == '__main__':
    print('Hello Ark')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/render_background.conf')
    parser.add_argument('--exp_name', type=str, default='render_background')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='cat') 
    parser.add_argument('--x_fov', type=float, default=None)

    args = parser.parse_args()

    
    f = open(args.conf)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', args.case)
    f.close()
    
    conf = ConfigFactory.parse_string(conf_text)
    conf['conf_path'] = args.conf
    conf['dataset.data_dir'] = conf['dataset.data_dir'].replace('CASE_NAME', args.case)
    print("Deal case {}".format(args.case))
    
    
    torch.cuda.set_device(args.gpu)
    exper = Exper(conf, args.exp_name, args.x_fov)
    exper.render_envmap()
    if exper.fov is not None:
        exper.render_view_background()
    else:
        exper.render_view_background_blend()

