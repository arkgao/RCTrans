import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import time


# This function is borrowed from IDR: https://github.com/lioryariv/idr

# pose is the camera-to-world matrix
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    t = (t[:3] / t[3])[:, 0]
    if K[2,2]<0:
        K[:,2] = -K[:,2]
        R[2,:] = -R[2,:]
        t = -np.dot(R,np.dot(np.linalg.inv(K),P[:,3:4]))[:,0]
    
    # K = K / K[2, 2]
    assert K[0,0]>0
    assert K[1,1]>0
    assert K[2,2]>0
    K = K / K[2,2]
    # assert K[2,2]==1.0
    
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = t

    return intrinsics, pose

# borrowed from NeuS
class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        start_time = time.time()
        self.device = torch.device('cuda')
        self.conf = conf
        
        
        img_idx = conf.get_list('img_idx',default=None)
        
        self.data_dir = conf.get_string('data_dir')
        print('Load data from {}'.format(self.data_dir))

        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name),allow_pickle=True)
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        
        def update_list(old_list, indices):
            return [ old_list[i] for i in indices]
        if img_idx is not None:
            self.images_lis = update_list(self.images_lis,img_idx)
        else:
            img_idx = np.arange(0,len(self.images_lis))
        self.n_images = len(self.images_lis)
        
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in img_idx]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in img_idx]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)    
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W


        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
        
        self.bb_mask = [None for _ in range(self.n_images)]

        # different from my previous method, now the transparent object is asumed to be inside a unit box
        self.bounding_box = torch.tensor([[-1.0, -1.0, -1.0],[1.0, 1.0, 1.0]], device=self.device)

        self.masks = None
        print('Load data: {} images End'.format(self.n_images))
        print('Time cost: {:.2f} mins'.format((time.time()-start_time)/60))
        print('-----------------------------------\n')
        
    
    def load_mask(self, mask_dir):
        """
            load pred mask for init_shape and optim_transparent
        Args:
            mask_dir (str): the path of pred mask
        """
        self.masks_lis = sorted(glob(mask_dir+'/*.png'))
        assert len(self.masks_lis)==len(self.images_lis), 'The num of masks mismatch the num of images'
        self.masks_np = np.stack([(cv.imread(mask_name)==255)[:,:,0:1] for mask_name in self.masks_lis])
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 1]
        print('Load mask from {}'.format(mask_dir))
        
    def load_margin(self, margin_dir):
        """load pred margin for optim_transparent
        Args:
            margin_dir (str): the path of pred margin
        """
        self.margin_lis = sorted(glob(margin_dir+'/*.png'))
        assert len(self.margin_lis)==len(self.images_lis), 'The num of masks mismatch the num of images'
        self.margin_np = np.stack([(cv.imread(margin_name)==255)[:,:,0:1] for margin_name in self.margin_lis])
        self.margins = torch.from_numpy(self.margin_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 1]
        print('Load margins from {}'.format(margin_dir))
        
        margin_pool = []
        for idx in range(len(self.margin_lis)):
            margin_sample = self.margins[idx,:,:,0].reshape(-1).bool()
            usample, vsample = torch.meshgrid(torch.linspace(0,self.W-1,self.W,device=self.device),torch.linspace(0,self.H-1,self.H,device=self.device),indexing='xy')
            pixels_x = usample.reshape(-1)[margin_sample]
            pixels_y = vsample.reshape(-1)[margin_sample]
            mask = self.masks[idx,:,:,0].reshape(-1)[margin_sample]
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()
            p = torch.matmul(self.intrinsics_all_inv[idx, None, :3, :3], p[:, :, None]).squeeze()
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
            rays_v = torch.matmul(self.pose_all[idx, None, :3, :3], rays_v[:, :, None]).squeeze()
            rays_o = self.pose_all[idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
            
            margin_pool.append(torch.cat([rays_o, rays_v, mask[:,None]], dim=-1))
        self.margin_pool = torch.vstack(margin_pool)
        
    def load_outdir(self,dir_folder):
        """
            load ray out direction for optim_transparent
        Args:
            dir_folder (str): the path of out_dir
        """
        self.out_dir = torch.from_numpy(np.load(os.path.join(dir_folder,'out_dir.npy'))).float().to(self.device)
        self.valid_mask = torch.from_numpy(np.load(os.path.join(dir_folder,'valid_mask.npy'))).float().to(self.device)
        assert self.out_dir.shape[0]==len(self.images_lis), 'The num of out_dir mismatch the num of images'
        assert self.valid_mask.shape[0]==len(self.images_lis), 'The num of valid_mask mismatch the num of images'
        print('Load out_dir from {}'.format(dir_folder))

       
    def load_pred_bounding_box(self,bb_box_path):
        """
            load pred bounding box for optim_transparent
        Args:
            bb_box_path (str): the path of pred bounding box
        """
        bb_box = np.load(bb_box_path)
        center = (bb_box[0,:]+bb_box[1,:]) / 2
        rad = (bb_box[1,:] - bb_box[0,:]) / 2
        rad[0:2] = 1.2*rad[0:2]
        bounds = np.vstack([center-rad, center+rad])
        self.bounding_box = torch.from_numpy(bounds).to(self.device)


    
            
    def get_sphere_intersection(self, rays_o, rays_d, r = 1.0):
        # Input: n_images x 4 x 4 ; n_images x n_rays x 3
        # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays
        
        origin_vector = -rays_o
        ray_dot = (rays_d * origin_vector).sum(dim=1,keepdim=True)
        under_sqrt = r ** 2 - (origin_vector.norm(2,1, keepdim=True) ** 2 - ray_dot ** 2)

        mask_intersect = under_sqrt > 1e-3
        
        near = torch.zeros_like(rays_o)[:,0:1]
        far  = torch.zeros_like(rays_d)[:,0:1]
        near = ray_dot - torch.sqrt(under_sqrt)
        near = near.clamp(min=1e-4)
        far = ray_dot + torch.sqrt(under_sqrt)

        far[far<0] = 2
        
        return mask_intersect.squeeze(), near, far
    
    
    def gen_rays_at(self, img_idx, resolution_level=1):
        """
            Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')  # i,j because the following ignore the transpose 
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def set_xfov(self,x_fov):
        """
            set the x fov of the camera for rendering backgroung images
        Args:
            x_fov (float): the x fov to set
        """
        focal = self.W / 2 / np.tan(x_fov/2*np.pi/180.0)
        self.intrinsics_all[:,0,0] = focal
        self.intrinsics_all[:,1,1] = focal
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        print('Set x fov to {}'.format(x_fov))

    def intersect_box(self, rays_o, rays_d):
        
        t_min = (self.bounding_box[0:1,:] - rays_o) / rays_d
        t_max = (self.bounding_box[1:2,:] - rays_o) / rays_d 
        t1 = torch.minimum(t_min,t_max)
        t2 = torch.maximum(t_min,t_max)
        t_near,_ = torch.max(t1,dim=1)
        t_far,_ = torch.min(t2, dim=1)
        return t_near < t_far

    # a potential problem is that the number of valid rays inside the mask is not enough for a batch
    # but it does not happen in current data
    def gen_random_inside_rays_at(self, img_idx, batch_size, with_outdir=False, with_normal=False):
        """
            pick random rays inside the mask
        """
        
        usample, vsample = torch.meshgrid(torch.linspace(0,self.W-1,self.W,device=self.device),torch.linspace(0,self.H-1,self.H,device=self.device),indexing='xy')
        mask = self.masks[img_idx,:,:,0].reshape(-1).bool()
        usample = usample.reshape(-1)[mask]
        vsample = vsample.reshape(-1)[mask]
        
        pix_idx = torch.randint(low=0, high=usample.shape[0], size=[batch_size],device=self.device)
        pixels_x = usample[pix_idx].long()
        pixels_y = vsample[pix_idx].long()
        
        color = self.images[img_idx][pixels_y, pixels_x,:]    # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
        if with_outdir:
            outdir = self.out_dir[img_idx][pixels_y,pixels_x,:]  # batch_size, 3
            valid_mask = self.valid_mask[img_idx][pixels_y,pixels_x].reshape(-1, 1) # batch_size, 1
            if with_normal:
                normal = self.normals[img_idx][pixels_y, pixels_x,:]  # batch_sieze, 3
                ray_data = torch.cat([rays_o, rays_v, color, normal, outdir, valid_mask], dim=-1)
            else:
                ray_data = torch.cat([rays_o, rays_v, color, outdir, valid_mask], dim=-1)
        else:
            ray_data = torch.cat([rays_o, rays_v, color], dim=-1)
        uv_data = torch.stack([pixels_y,pixels_x],dim=-1)
        
        
        return ray_data, uv_data
    
    def gen_random_margin(self, batch_size):
        pix_idx = torch.randint(low=0, high=self.margin_pool.shape[0], size=[batch_size],device=self.device)
        data = self.margin_pool[pix_idx]
        return data
        
    
    def gen_random_rays_at(self, img_idx, batch_size, with_mask = False):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        if not with_mask:
            return torch.cat([rays_o, rays_v, color], dim=-1)   # batch_size, 9
        else:
            mask = self.masks[img_idx][(pixels_y, pixels_x)].float()
            return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1)   # batch_size, 10
        
    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    def image_at(self, idx, resolution_level=1):
        # img = cv.imread(self.images_lis[idx])
        img = self.images_np[idx]   # [0,1]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 1)

    def mask_at(self, idx, resolution_level=1):
        # mask = cv.imread(self.masks_lis[idx])
        mask = self.masks_np[idx].astype(np.float32) # bool
        return (cv.resize(mask, (self.W // resolution_level, self.H // resolution_level)))
