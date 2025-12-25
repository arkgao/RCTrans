import os
import random
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.RayTracer import RayTracer
from models.utils import enlarge_bounding_box

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Runner:
    """
        Initialize the object shape with masks and fit the SDF network.
        Basicly, the training process is the same as NeuS, but with only mask loss and eikonal loss.
        And it takes less iterations.
    """
    def __init__(self, conf_path, case='CASE_NAME', exp_name='init_shape'):
        self.device = torch.device('cuda')
        self.exp_name = exp_name

        # Configuration
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
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight', default=1)
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.obj_sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.obj_deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        
        params_to_train += list(self.obj_sdf_network.parameters())
        params_to_train += list(self.obj_deviation_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)


        if os.path.exists(os.path.join(self.conf['dataset.data_dir'],'min_z.txt')):
            self.min_z = float(np.loadtxt(os.path.join(self.conf['dataset.data_dir'],'min_z.txt')))
            print('add extra constraint:  min_z=',self.min_z)
        else:
            self.min_z = None
        
        self.tracer = RayTracer(self.obj_sdf_network,
                                self.obj_deviation_network,
                                obj_box = None,
                                **self.conf['model.tracer'])


        # Load estimated mask in stage1
        if self.conf.get_string('mask_dir', default = None):
            mask_dir = self.conf.get_string('mask_dir')
        else:    
            mask_dir = os.path.join(self.base_exp_dir, 'export_mask','mask')
            
        self.dataset.load_mask(mask_dir)
        print('Load mask from {}'.format(mask_dir))    
        
        self.base_exp_dir = os.path.join(self.base_exp_dir, self.exp_name) 
        
        self.file_backup()
        

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size, with_mask=True)

            rays_o, rays_d, _, mask_gt = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            
            mask_gt_sum = mask_gt.sum() + 1e-5
            
            render_out = self.tracer.ray_tracing(rays_o, rays_d)

            s_val = render_out['s_val']
            gradient_error = render_out['gradient_error']
            weight_max = torch.max(render_out['weights'], dim=-1, keepdim=True)[0]
            weight_sum = render_out['weights_sum']

            eikonal_loss = gradient_error.mean()
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask_gt)
            
            loss = mask_loss * self.mask_weight + eikonal_loss  * self.igr_weight
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            
            self.writer.add_scalar('Static/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.iter_step)
            self.writer.add_scalar('Static/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Static/weight_max', (weight_max * mask_gt).sum() / mask_gt_sum, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                pred_point = self.validate_mesh()
                
            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
        
        self.validate_detail_mesh()
        
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def save_checkpoint(self):
        checkpoint = {
            'obj_network': self.obj_sdf_network.state_dict(),
            'obj_deviation_network' : self.obj_deviation_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
    def find_latest_ckpt(self,path):
        model_list = sorted(glob(os.path.join(path,'*.pth')))
        return model_list[-1]

    def load_checkpoint(self):
        model_path = self.find_latest_ckpt(os.path.join(self.base_exp_dir,'checkpoints'))
        logging.info('Load trained stage1 model from: {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.obj_sdf_network.load_state_dict(checkpoint['obj_network'])
        self.obj_deviation_network.load_state_dict(checkpoint['obj_deviation_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

    def validate_image(self, idx=-1, resolution_level=4):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape        
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):

            render_out = self.tracer.ray_tracing(rays_o_batch,
                                              rays_d_batch)
            normal = render_out['normal'].detach().cpu().numpy()
            sphere_mask = render_out['sphere_mask'].detach().cpu().numpy()
            mask = render_out['weights_sum'].squeeze(1).detach().cpu().numpy() > 0.1
            mask = mask & sphere_mask
            normal[~mask] = 0
            out_normal_fine.append(normal)
            del render_out
            
        normal_img = np.concatenate(out_normal_fine, axis=0)
        rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
        normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                        .reshape([H, W, 3]) * 128 + 128).clip(0, 255)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        cv.imwrite(os.path.join(self.base_exp_dir,'normals','{:0>8d}_{}.png'.format(self.iter_step, idx)),
                           normal_img)

    def validate_mesh(self, resolution=256, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        scale = self.dataset.scale_mats_np[0][0,0]
        vertices, triangles =\
            self.tracer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        mesh = trimesh.Trimesh(vertices, triangles)
        self.bounds = mesh.bounds
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        return torch.tensor(np.array(mesh.vertices),device='cuda')
    
    def validate_detail_mesh(self, resolution = 512, threshold=0.0):
        bounds = enlarge_bounding_box(self.bounds)
        bound_min = torch.from_numpy(bounds[0,:])
        bound_max = torch.from_numpy(bounds[1,:])
        vertices, triangles =\
            self.tracer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        mesh = trimesh.Trimesh(vertices, triangles)
        self.bounds = mesh.bounds
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'final.ply'.format(self.iter_step)))
        # we save the bounding box for following optimization step
        np.save(os.path.join(self.base_exp_dir, 'bounding_box.npy'), self.bounds)
        return os.path.join(self.base_exp_dir, 'meshes', 'final.ply'.format(self.iter_step))
        
if __name__ == '__main__':
    print('Hello Ark')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/init_shape.conf')    
    parser.add_argument('--exp_name', type=str, default='init_shape')
    parser.add_argument('--extra_name', type=str, default='')    
    parser.add_argument('--gpu', type=int, default=0)                       
    parser.add_argument('--case', type=str, default='cat')                   

    args = parser.parse_args()
    if args.extra_name:
        args.exp_name += '_{}'.format(args.extra_name)
     
    print("Deal case {}".format(args.case))
    
    torch.cuda.set_device(args.gpu)

    runner = Runner(args.conf, args.case, args.exp_name)      
    runner.train()
