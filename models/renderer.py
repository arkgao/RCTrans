import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import extract_geometry, sample_pdf,  near_far_from_sphere
from torchvision import transforms
import kaolin as kal
import cv2

# The original NeuS renderer
class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 obj_box=None,
                 test_mode=False):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.obj_box = obj_box
        self.test_mode=test_mode

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def get_bb_weights(self, pts, inverse_mode = True, beta=400):
        """
            Get the weight of bounding box
        Args:
            pts ([samples, 3]) : sample point
            bounding_box_val ([2,3]) : dim0: min, max; dim1: x, y, z
            inverse_mode (bool)      : False: weights inside box is 1; True: weights inside box is 0
            beta (int, optional)     : control the strict

        Returns:
            weights ([samples, 1]): the weights considering bounding box
        """
        
        center  = (self.obj_box[0,:]+self.obj_box[1,:]) / 2 
        rad = (self.obj_box[1,:] - self.obj_box[0,:]) / 2

        x_dist = torch.abs(pts[...,0:1] - center[0])
        y_dist = torch.abs(pts[...,1:2] - center[1])
        z_dist = torch.abs(pts[...,2:3] - center[2])

        weights = torch.sigmoid(beta*(rad[0]-x_dist))*torch.sigmoid(beta*(rad[1]-y_dist))*torch.sigmoid(beta*(rad[2]-z_dist))
        weights[weights>=0.5] = 1
        weights[weights<0.5] = 0
        
        if inverse_mode:
            return 1.0 - weights
        else:
            return weights

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    sample_dir = None,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        if sample_dir is not None:
            dirs = sample_dir[:,None,:].expand(pts.shape)
        else:
            dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        
        #todo maybe should use bounding box to guide the sample, instead of weight the sample after
        if self.obj_box is not None:
            bb_weights = self.get_bb_weights(pts, inverse_mode=True).reshape(batch_size, n_samples)
            hit_obj = ((1-bb_weights) * alpha).sum(-1)> 0.5
            alpha = bb_weights * alpha
        else:
            hit_obj = None     

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        output = {
            'color': color,
            'weights': weights,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'hit_obj': hit_obj,
            's_val': 1.0 / inv_s,
            'weights_sum': weights_sum,
            'gradient_error': gradient_error,
            'inside': inside_sphere,
        }
        if self.test_mode:
            output.update({
                'sdf': sdf,
                'dists': dists,
                'alpha': alpha,
                'mid_z_vals': mid_z_vals,
                'pdf': p.reshape(batch_size, n_samples),
                'cdf': c.reshape(batch_size, n_samples),
            })
        return output

    def render(self, rays_o, rays_d, sample_dir = None, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        near, far = near_far_from_sphere(rays_o,rays_d)
        # _, near, far = get_sphere_intersection(rays_o,rays_d)
        
        batch_size = len(rays_o)
        sample_dist = (far-near) / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:      
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples   
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:     
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

        n_samples = self.n_samples + self.n_importance

        z_vals_feed = None
        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    sample_dir=sample_dir,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        output = {
            'color': ret_fine['color'],
            'weights': weights,
            'hit_obj': ret_fine['hit_obj'],            
            's_val': s_val,
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': ret_fine['gradients'],
            'gradient_error': ret_fine['gradient_error'],
            'inside': ret_fine['inside'],
        }
        if self.test_mode:
            output.update({
                
                'z_val' : z_vals,
                'z_vals_feed': z_vals_feed,
                'cdf': ret_fine['cdf'],
                'sdf': ret_fine['sdf'],
                'pdf': ret_fine['pdf'],
                'alpha': ret_fine['alpha']
                })    
        return output
        

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))


def gaussian_blur(map, sigma, kernel_size=-1):
    if kernel_size == -1:
        kernel_size = max(int(3*sigma*2)//2*2+1,3)
    return transforms.functional.gaussian_blur(map, kernel_size, sigma)

class TextureRenderer():
    """
        The texture renderer for rendering the transparent object with the scene texture
    """
    def __init__(self):
        self.sigma = 1
    
    def set_texture_map(self, texture):
        self.texture_map = texture
        if len(self.texture_map) == 3:
            self.texture_map.unsqueeze(0)
        self.current_texture = gaussian_blur(self.texture_map, self.sigma)
    
    def set_gaussian_sigma(self, sigma):
        self.sigma = sigma
        self.current_texture = gaussian_blur(self.texture_map, self.sigma)
    
    def texture_mapping(self,texture_coords):
        texture_coords[:,1] = 1 - texture_coords[:,1]
        texture_coords = texture_coords.unsqueeze(0)
        image = kal.render.mesh.texture_mapping(texture_coords,
                                        self.current_texture, 
                                        mode='bilinear')
        return image.squeeze(0)
    
    def render(self, rays_o, rays_d):
        """the core render

        Args:
            rays_o [num, 3]: 
            rays_d [num, 3]: _description_
            cos_anneal_ratio (float, optional): _description_. Defaults to 0.0.

        Returns:
            color  [num, 3]: the same num as rays
            weight_sum(mask): whether the ray hit the scene 
        """
        k = -rays_o[:,2] / (rays_d[:,2]+1e-4)
        x = rays_o[:,0] + k * rays_d[:,0]
        y = rays_o[:,1] + k * rays_d[:,1]
        u = (x + 1.0) / 2 
        v = (y + 1.0) / 2
        texture_coords = torch.stack([u,v],-1)
        color = self.texture_mapping(texture_coords)
        invalid_mask = (u<0) | (u>1) | (v<0) | (v>1)
        color[invalid_mask,:] = 0
        
        weight_sum = torch.ones_like(rays_o)[:,0:1]
        dist = torch.square(x) + torch.square(y)
        weight_sum[dist>1,:] = 0
        render_output = {
            'color': color,
            'weight_sum': weight_sum,
            "intersection":torch.stack([x,y],-1)
        }
        
        return render_output
    
class EnvRenderer():
    """
        Render the transparent object's color using environment map
    """
    def __init__(self):
        self.sigma = 1

    def set_envmap(self, envmap):
        self.envmap = envmap
        if len(self.envmap) == 3:
            self.envmap.unsqueeze(0)
        self.current_envmap = gaussian_blur(self.envmap, self.sigma)
    
    def set_gaussian_sigma(self, sigma):
        self.sigma = sigma
        self.current_envmap = gaussian_blur(self.envmap, self.sigma)
    
    def spherical_mapping(self, ray_dirs):
        '''
        Calculate the u,v according to the ray_dir using spherical_mapping
        Dims: ray_dirs: N*3
        '''
        x = ray_dirs[:, 0]
        y = ray_dirs[:, 1]
        z = ray_dirs[:, 2]
        
        # mapping x,y,z to u,v([0, 1]), mention the Mitsuba Coordinate and rotation
        u = 0.5 + torch.atan2(-x, -y) / (2 * torch.pi)
        v = 0.5 - torch.asin(z.clamp(min=-1+1e-6, max=1-1e-6)) / torch.pi
        
        return u.clip(0, 1), v.clip(0, 1)
    
    def render(self, rays_o, rays_d):
        '''use envmap to render color
        dims: rays_o: batch_size * 3
              rays_d: batch_size * 3
        '''
        N = rays_d.shape[0] # batch size
        
        u, v = self.spherical_mapping(rays_d)
        # notice that kaolin uses OpenGL coordinate, y axis is from bottom to top, change v to 1 - v
        v = 1 - v
        
        grid = torch.stack([u, v], dim=-1).reshape(1, -1, 2) # stack the u,v coordinates
        colors = kal.render.mesh.texture_mapping(grid, self.current_envmap, mode='bilinear')
        colors = colors.reshape(N, 3)

        render_output = {
            'color': colors,
        }
        return render_output
    

class BackgroundRenderer():
    """
        Render the background outside the unit sphere
        It can use nerf++ or environment map to render the background
    """
    def __init__(self, nerf_outside=None, envmap=None):
        self.nerf_outside = nerf_outside
        self.envmap_path = envmap
        self.n_samples = 64

        if envmap is not None:
            self.set_envmap()

    def set_envmap(self):
        img = cv2.imread(self.envmap_path)
        img = img / 255.0
        gamma = 2.2
        tonemap_img = lambda x: np.power(x, gamma)
        self.envmap = torch.from_numpy(tonemap_img(img.clip(0.0, 1.0))).float().cuda().unsqueeze(0)
        # self.envmap = torch.from_numpy(img).float().cuda().unsqueeze(0)

    def render(self, rays_o, rays_d):
        if self.nerf_outside is not None:
            result = self.render_with_nerfoutside(rays_o,rays_d)
        else:
            result = self.render_with_envmap(rays_o,rays_d)

        return result

    def render_with_nerfoutside(self, rays_o, rays_d):
        '''use nerf++ to render the background
        dims: rays_o: batch_size * 3
              rays_d: batch_size * 3
        '''

        # Define near and far bounds for the rays
        near, far = near_far_from_sphere(rays_o, rays_d)
        batch_size = len(rays_o)
        
        # Sample points along the rays
        sample_dist = (far-near) / self.n_samples  # Assuming the region of interest is a unit sphere(inverse)

        # Calculate z_vals for NeRF++
        z_vals = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_samples + 1.0), self.n_samples)
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand([batch_size, z_vals.shape[-1]])
        z_vals = lower[None, :] + (upper - lower)[None, :] * t_rand
        z_vals = far / torch.flip(z_vals, dims=[-1]) + 1.0 / self.n_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]  # batch_size, n_samples-1
        dists = torch.cat([dists, sample_dist], -1) # batch_size, n_samples

        # Compute the sampled points
        mid_z_vals = z_vals + dists * 0.5
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        # Normalize points for NeRF++
        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # batch_size, n_samples, 4
        pts = pts.reshape(-1, 4)
        dirs = rays_d[:, None, :].expand(batch_size, self.n_samples, 3).reshape(-1, 3)

        density, sampled_color = self.nerf_outside(pts, dirs)

        # Compute alpha and weights
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, self.n_samples)) * dists)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1.0 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]

        # Compute final color
        sampled_color = sampled_color.reshape(batch_size, self.n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        depth = (weights * mid_z_vals).sum(dim=-1, keepdim=True)

        result = {
            'color_fine': color,
            'depth': depth,
            'weights': weights,
            'weight_sum': weights.sum(dim=-1, keepdim=True),
        }
        return result

    def spherical_mapping(self, ray_dirs):
        '''
        Calculate the u,v according to the ray_dir using spherical_mapping
        Dims: ray_dirs: N*3
        '''
        x = ray_dirs[:, 0]
        y = ray_dirs[:, 1]
        z = ray_dirs[:, 2]
        
        # mapping x,y,z to u,v([0, 1]), mention the Mitsuba Coordinate and rotation
        u = 0.5 + torch.atan2(-x, -y) / (2 * torch.pi)
        v = 0.5 - torch.asin(z.clamp(min=-1+1e-6, max=1-1e-6)) / torch.pi
        
        return u.clip(0, 1), v.clip(0, 1)

    def render_with_envmap(self, rays_o, rays_d):
        '''use envmap to render the background
        dims: rays_o: batch_size * 3
              rays_d: batch_size * 3
        '''
        N = rays_d.shape[0] # batch size
        
        u, v = self.spherical_mapping(rays_d)
        # notice that kaolin uses OpenGL coordinate, y axis is from bottom to top, change v to 1 - v
        v = 1 - v
        
        grid = torch.stack([u, v], dim=-1).reshape(1, -1, 2) # stack the u,v coordinates
        envmap = self.envmap.permute(0, 3, 1, 2) 
        colors = kal.render.mesh.texture_mapping(grid, envmap, mode='bilinear')
        colors = colors.reshape(N, 3)

        gamma = 2.2
        tonemap_img = lambda x: torch.pow(x, 1./gamma)
        colors = tonemap_img(colors)

        result = {
            'color_fine': colors,
            'depth': None,
        }
        return result
