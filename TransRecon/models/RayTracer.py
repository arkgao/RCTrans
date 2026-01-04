import torch
import torch.nn.functional as F
import numpy as np
import mcubes
from models.utils import get_sphere_intersection, get_cuboid_intersection


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs,indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles



class RayTracer():
    """
        The ray tracing caculate the geometry information without any color,
        including two ways: the volume+sum way in NeuS, and sample+interpolate way in Geo-Neus
    """
    def __init__(self, sdf_network, deviation_network, obj_box,
                 n_samples, n_importance, up_sample_steps, perturb,
                 test_mode=False, ray_tracing_method='volume', sample_full=False):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.obj_box = obj_box  # [[min], [max]] 2x3
        if self.obj_box is not None:
            self.obj_aabb = torch.cat([obj_box[0], obj_box[1]]) # [min, max] 6
        self.test_mode = test_mode
        self.ray_tracing_method = ray_tracing_method
        self.sample_full = sample_full

    def remove_boundingbox(self):
        self.obj_box = None
    
    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, inverse_mode=False):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        # assuming the object is inside the unit box
        inside_box = ( (pts[:,:-1,0].abs()<1) & (pts[:,:-1,1].abs()<1) & (pts[:,:-1,2].abs()<1)) | ((pts[:,1:,0].abs()<1) & (pts[:,1:,1].abs()<1) & (pts[:,1:,2].abs()<1))
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
        cos_val = cos_val.clip(-1e3, 0.0) * inside_box

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        if inverse_mode:
            alpha = 1 - alpha
        
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
  
    # This function is from eikonal field https://github.com/m-bemana/eikonalfield
    def get_bb_weights(self, pts, inverse_mode = True, beta=400):
        """
            Get the weight of bounding box. Remove weights outside the bounding box
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
 
    def ray_tracing_interpolate_core(self,
                                rays_o,
                                rays_d,
                                z_vals,
                                sample_dist):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]

        # borrowed from geo-neus
        sdf_d = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        sign = prev_sdf * next_sdf
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
        idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
        tmp = torch.einsum("ab,b->ab", (sign, idx))
        prev_idx = torch.argmax(tmp, 1, keepdim=True)
        next_idx = prev_idx + 1
        
        sdf1 = torch.gather(sdf_d, 1, prev_idx)
        sdf2 = torch.gather(sdf_d, 1, next_idx)
        z_vals1 = torch.gather(mid_z_vals, 1, prev_idx)
        z_vals2 = torch.gather(mid_z_vals, 1, next_idx)
        z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        pts_sdf0 = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_sdf0[..., :, None]  # [batch_size, 1, 3]
        
        gradients_sdf0 = self.sdf_network.gradient(pts_sdf0.reshape(-1, 3)).squeeze().reshape(batch_size, 3)
        depth = z_vals_sdf0
        normal = gradients_sdf0 / (gradients_sdf0.norm(2,1,keepdim=True) + 1e-5)
        gradient_error = (torch.linalg.norm(gradients_sdf0, ord=2, dim=-1) - 1.0) ** 2
        max_z_val,_ = torch.max(z_vals,dim=1)
        hit_mask = (z_vals_sdf0.squeeze(1) < max_z_val) & (z_vals_sdf0.squeeze(1) > 0) & (tmp.sum(1)>0.0)

        hit_mask = hit_mask.reshape(-1)
        
        assert not (torch.isnan(normal).any())

        output = {
                'depth': depth,
                'normal': normal,
                'weights_sum': None,
                'gradient_error': gradient_error,
                'hit_mask': hit_mask,
                's_val': None,
        }
        return output
 
    def ray_tracing_volume_core(self,
                        rays_o,
                        rays_d,
                        z_vals,
                        sample_dist,
                        inverse_mode = True,
                        cos_anneal_ratio=1.0):
        """ The core of ray tracing in sdf neural field using volumn rendering

        Args:
            rays_o (_type_): _description_
            rays_d (_type_): _description_
            z_vals (_type_): _description_
            sample_dist (_type_): _description_
            inverse_mode (bool, optional): _description_. Defaults to True.
            cal_grad (bool, optional): _description_. Defaults to True.
            cos_anneal_ratio (float, optional): _description_. Defaults to 1.0.
        """
        batch_size, n_samples = z_vals.shape
        
        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]

        gradients = self.sdf_network.gradient(pts).squeeze()
        
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        
        if inverse_mode:
            alpha = 1 - alpha

        # For the inner ray tracing, no need to compute the bb weight
        if (not inverse_mode) and (not self.obj_box is None):
            bb_weights = self.get_bb_weights(pts, inverse_mode=False).reshape(batch_size, n_samples)
            alpha = bb_weights * alpha 


        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        # inside_sphere = (pts_norm < 1.0).float().detach()
        inside_box = (pts[:,0] > -1) & (pts[:,0] < 1) & (pts[:,1] > -1) & (pts[:,1] < 1) & (pts[:,2] > -1) & (pts[:,2] < 1)
        inside_box = inside_box.reshape(batch_size, n_samples).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
                
        # Eikonal loss
        gradients = gradients.reshape(batch_size,n_samples,3)
        gradient_error = (torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2

        depth = (weights * z_vals).sum(dim=1,keepdim=True)
        normal = (weights[:, :, None] * gradients).sum(dim=1)
        normal = normal / (normal.norm(2,1,keepdim=True)+1e-5)
        gradient_error = gradient_error.mean(1)
                
        assert (not torch.isnan(normal).any())
        assert (not torch.isnan(depth).any())
        assert (not torch.isnan(gradient_error).any())
        
        weights_sum = weights.sum(dim=-1,keepdim=True)

        output = {
                'depth': depth,
                'normal': normal,
                'weights': weights,
                'weights_sum': weights_sum,
                'gradient_error': gradient_error,
                'inside_sphere': inside_box,
                's_val': 1.0 / inv_s,
                'hit_mask': weights_sum.squeeze(1)>0.1
            }
       
        if self.test_mode:
            output.update({
                'sdf': sdf,
                'alpha': alpha,
                'dists': dists,
                'mid_z_vals': mid_z_vals,
                'pdf': p.reshape(batch_size, n_samples),
                'cdf': c.reshape(batch_size, n_samples),
                })
        
        return output
    
    def ray_tracing(self, rays_o, rays_d, inverse_mode=False, tracing_method=None):
        """trace the ray for the first hitted point on the surface with volume+sum way (NeuS) or sample+interpolate way (Geo-Neus)

        Args:
            rays_o (torch.tensor): [batch, 3]
            rays_d (torch.tensor): [batch, 3]
            inverse_mode (bool, optional): whether tracing ray from inside to outside. Defaults to False.

        Returns:
            normal: normal of the hitted point
            depth : depth of the hitted point
            mid_result: other values for extra supervision and debug
        """
        if self.sample_full:
            sphere_mask, near, far = get_sphere_intersection(rays_o, rays_d)
        else:
            # sample the unit cube, more robust
            sphere_mask, near, far = get_cuboid_intersection(rays_o, rays_d,self.obj_aabb)

        if tracing_method == None:
            tracing_method = self.ray_tracing_method

        batch_size = len(rays_o)
        sample_dist = (far - near) / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]
        
        n_samples = self.n_samples   
        perturb = self.perturb
        
        if perturb > 0:   
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        
        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples)

                for i in range(self.up_sample_steps):  
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i,
                                                inverse_mode)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))
        n_samples = self.n_samples + self.n_importance
        
        if tracing_method == 'volume':
            ret_fine = self.ray_tracing_volume_core(rays_o,
                                        rays_d,
                                        z_vals,
                                        sample_dist,
                                        inverse_mode = inverse_mode,
                                        cos_anneal_ratio=1.0)
            output = {
            'sphere_mask' : sphere_mask,
            'gradient_error' : ret_fine['gradient_error'],
            'normal' : ret_fine['normal'],
            'depth' : ret_fine['depth'],
            'weights_sum' : ret_fine['weights_sum'],
            'hit_mask': ret_fine['hit_mask'],
            's_val': ret_fine['s_val'],
            'weights': ret_fine['weights']
            }

        elif tracing_method == 'interpolate':
            ret_fine = self.ray_tracing_interpolate_core(rays_o,
                                        rays_d,
                                        z_vals,
                                        sample_dist,
                                        )
            output = {
            'sphere_mask' : sphere_mask,
            'gradient_error' : ret_fine['gradient_error'],
            'normal' : ret_fine['normal'],
            'depth' : ret_fine['depth'],
            'hit_mask': ret_fine['hit_mask'],
            }
        else:
            assert False
        

        

        
        if self.test_mode:
            output.update({
                            'weights' : ret_fine['weights'],
                            's_val' : ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True),
                            'z_vals' : z_vals,
                            'sdf' : ret_fine['sdf'],
                            'alpha': ret_fine['alpha'],
                            'p' : ret_fine['pdf'],
                            'c' : ret_fine['cdf'],
                            })
        return output

    
    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
        
    
    