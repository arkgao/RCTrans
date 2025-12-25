import torch
from models.utils import dot, extract_geometry
from models.RayTracer import RayTracer


class TransRender():
    def __init__(self,
                 tracer:RayTracer,
                 renderer,
                 objIOR = 1.4723,
                 airIOR = 1.0,
                 reflection_flag = True):
        """  
            Render a transparent object within a scene
        Args:
            tracer: get the ray surface intersection of the transparent object
            renderer: to render the scene color
            objIOR: the init index of refraction of the object.
            airIOR: the index of refraction of air.
            reflection_flag: whether to render reflection.
        """
        
        # object
        self.tracer = tracer 
        # scene
        self.renderer = renderer
        self.objIOR = torch.tensor(objIOR).to(next(self.tracer.sdf_network.parameters()).device)
        self.airIOR = torch.tensor(airIOR).to(next(self.tracer.sdf_network.parameters()).device)
        
        self.reflection_flag = reflection_flag
        
    def refract_dir(self,rays_dir,normal,relative_IOR):
        """compute the new direction of refracted rays

        Args:
            rays_dir (torch.Tensor): [batch, 3]     the incomling light direction (from a distance to the surface)
            normal (torch.Tensor): [batch, 3]       the surface normal
            relative_IOR (torch.Tensor): [batch, 1] relative index of refraction (intIOR / extIOR)
            
        Returns:
            out_dir (torch.Tensor): [batch, 3]      the out light direction (from the surface to a distance)
            total_reflect: [batch,1]                whether the total reflection occurs
        
        """ 
        inv_rela_IOR = 1 / relative_IOR
        cos_theta_i = dot(-rays_dir, normal, True).clip(1e-5,1)
        sin2_theta_i = (1 - cos_theta_i **2 ).clamp(min = 0)
        sin2_theta_t = inv_rela_IOR **2 * sin2_theta_i
        total_reflect = (sin2_theta_t >= 1)
        cos_theta_t = torch.sqrt(1 - sin2_theta_t.clamp(max = 1-1e-5,min=1e-5))
        out_dir = inv_rela_IOR * rays_dir + (inv_rela_IOR * cos_theta_i - cos_theta_t) * normal

        # check the formula
        # assert(torch.all(torch.abs(torch.linalg.norm(out_dir,dim=1)-1)<1e-4))
        out_dir = out_dir / torch.linalg.norm(out_dir,dim=1,keepdim=True)
        
        return out_dir, total_reflect.squeeze(-1), cos_theta_i
    
    def frenel_coefficient(self, cos_theta_i, extIOR, intIOR):
        """compute the radiance coefficient of refraction and reflection
           accoring to the frenel law

        Returns:
            reflect_cof: the coefficient of reflection
            refract_cof: the coefficient of refraction
        """
        n1 = extIOR
        n2 = intIOR
        sin2_theta_i = 1 - cos_theta_i ** 2
        sin2_theta_t = (n1**2 / n2**2 * sin2_theta_i).clamp(max=1)
        rs = (n1*cos_theta_i - n2 * torch.sqrt(1 - sin2_theta_t)) \
            /  (n1*cos_theta_i + n2 * torch.sqrt(1 - sin2_theta_t))
        rp = (n1 * torch.sqrt(1 - sin2_theta_t) - n2*cos_theta_i ) \
            /  (n1 * torch.sqrt(1 - sin2_theta_t) + n2*cos_theta_i)

        Rs = rs**2
        Rp = rp**2
        reflect_cof = (Rs + Rp) / 2
        transimit_cof = 1 - reflect_cof
        
        # assert ((reflect_cof<1).all())
        
        return reflect_cof.detach(), transimit_cof.detach()
    
    def reflect_ray(self, rays_o, rays_d, depth, normal):
        """
            Specular Reflection
        Args:
            rays_o (torch.tensor): ray origin
            rays_d (torch.tensor): ray direction from far to the boundary
            depth (torch.tensor): depth of the intersection
            normal (torch.tensor): surface normal
        """
        #           normal        
        #   ray_dir   ^    new_dir
        #        \    |    / 
        #         \   |   / 
        #          \  |  /
        #           \ | /
        #            \|/
        #  -----------------------------------   
        
        cos_alpha = dot(-rays_d, normal, True)
        new_dir = 2*cos_alpha*normal + rays_d
        
        # assert(torch.all(dot(normal+1e-5,rays_d-1e-5)<0))    
        new_ori = rays_o + depth*rays_d

        # todo check the chiocs of 2e-2
        new_ori = new_ori + 2e-2*new_dir
        
        output = {
            'rays_o': new_ori,
            'rays_d': new_dir,
        }
        
        return output
    
    def refract_ray(self, rays_o, rays_d, depth, normal, inverse_mode = False):
        """ Refraction
        Args:
            rays_o (torch.tensor): ray origin
            rays_d (torch.tensor): ray direction from far to the boundary
            depth (torch.tensor): depth of the intersection
            normal (torch.tensor): surface normal
            inverse_mode (bool, optional): whether the ray is from inside to outside. Defaults to False.
        """
        # Refraction
        #    ray_dir  ^ normal
        #       `     |     
        #         `   |     
        #           ` |    extIOR
        #        -----|------------- 
        #             | \
        #             |  \   intIOR
        #             |   \
        #                   out_dir      
        # ----------------------------------------------------------------------------------------------------------
         
        if not inverse_mode:
            intIOR = self.objIOR
            extIOR = self.airIOR
        else:
            intIOR = self.airIOR
            extIOR = self.objIOR
            normal = -normal
            
        # assert(torch.all(dot(normal+1e-5,rays_d-1e-5)<0))    
        intIOR = intIOR * torch.ones_like(rays_o)[:,0:1]
        extIOR = extIOR * torch.ones_like(rays_o)[:,0:1]
        new_dir, total_reflect, cos_theta_i = self.refract_dir(rays_d, normal, intIOR / extIOR)
        reflect_cof, transimit_cof = self.frenel_coefficient(cos_theta_i, extIOR, intIOR)
        new_ori = rays_o + depth*rays_d

        new_ori = new_ori + 5e-2*new_dir
        
        assert (not torch.isnan(new_ori).any())
        assert (not torch.isnan(new_dir).any())
        assert (not torch.isnan(transimit_cof).any())
        
        output = {
            'rays_o': new_ori,
            'rays_d': new_dir,
            'totalRef': total_reflect,
            'ref_cof': reflect_cof,
            'trans_cof': transimit_cof,
        }
        
        return output
   
    def first_bounce(self, rays_o, rays_d):
        """ 
            The first bounce
            The ray hit the transparent object
            
            mask is determined by:   
                1.whether the ray hit the object (weight_sum),
                2.whether the ray is inside the unit sphere (acctually, it is not necessary)
        """
        # tracing
        inverse_mode = False
        tracing_output = self.tracer.ray_tracing(rays_o, rays_d, inverse_mode=inverse_mode)
        depth = tracing_output['depth']
        normal = tracing_output['normal']
        
        # mask
        sphere_mask = tracing_output['sphere_mask']
        hit_mask = tracing_output['hit_mask']
        mask = sphere_mask & hit_mask
        
        normal = normal[mask]
        depth = depth[mask]
        # result
        ref_ray = self.reflect_ray(rays_o[mask], rays_d[mask], depth, normal)
        trans_ray = self.refract_ray(rays_o[mask], rays_d[mask], depth, normal, inverse_mode=inverse_mode)       
        assert (not torch.any(trans_ray['totalRef']))
        ref_ray['ref_cof'] = trans_ray['ref_cof']
        
        return ref_ray, trans_ray, mask, tracing_output['gradient_error'][mask], normal

    def second_bounce(self,rays_o, rays_d):
        """ 
            The second bounce
            The ray shot out from the transparent object
            mask is determined by:   
                1. whether the ray hit the object (consider the small eta, maybe some ray would hit the object)
                2. whether the ray is totally reflected
        """
        
        # tracing
        inverse_mode = True
        tracing_output = self.tracer.ray_tracing(rays_o, rays_d, inverse_mode=inverse_mode)
        depth = tracing_output['depth']
        normal = tracing_output['normal']
    
        trans_ray = self.refract_ray(rays_o, rays_d, depth, normal, inverse_mode=inverse_mode)
        
        
        # mask
        end_point = rays_o + tracing_output['depth'] * rays_d
        unibody_mask = end_point[:,2] > -1
        
        totalRef_bounce2 = trans_ray['totalRef']
        totalRef_mask = ~totalRef_bounce2
        
        mask = unibody_mask & totalRef_mask

        
        # result
        ref_ray = {}
        
        new_trans_ray = {
            'rays_o': trans_ray['rays_o'][mask],
            'rays_d': trans_ray['rays_d'][mask],
            'trans_cof': trans_ray['trans_cof'][mask],
        }
        return ref_ray, new_trans_ray, mask, tracing_output['gradient_error'][mask]
    
    def third_rendering(self, rays_o, rays_d):
        """
            The third tracing, to render the color of the refracted ray
            mask is determined by:
                1. whether the ray hit the object again (refraction more than 2 times is ignored)
                2. whether the ray hit the plane
        """
        # third rendering
        render_output = self.renderer.render(rays_o, rays_d)
        tracing_output = self.tracer.ray_tracing(rays_o, rays_d)
        
        
        # mask
        twice_mask = ~tracing_output['hit_mask']
        mask = twice_mask
        
        # result
        color = render_output['color'][mask]
        
        return color, mask


      
    def render_transparent(self, rays_o, rays_d):
        """
        Render the transparent object 
        only render the rays that hit the transparent, others are set to 0
        """
        # First Bounce
        ref_ray, first_trans_ray, first_mask, grad_error_1, normal = self.first_bounce(rays_o,rays_d)
        hit_mask = first_mask.clone()
        
        # Second Bounce
        _, second_trans_ray, second_mask, grad_error_2 = self.second_bounce(first_trans_ray['rays_o'],
                                                              first_trans_ray['rays_d'])
        totalRef_mask = second_mask.clone()
        
        # third rendering
        trans_color, third_mask = self.third_rendering(second_trans_ray['rays_o'], second_trans_ray['rays_d'])
        valid_render_mask = third_mask.detach()        
        
        second_mask = second_mask.masked_scatter(second_mask,third_mask)
        first_mask = first_mask.masked_scatter(first_mask,second_mask)
        
        color_fine = torch.zeros_like(rays_o)
        color_fine[first_mask,:] =  (first_trans_ray['trans_cof'][second_mask,:] * \
                                    second_trans_ray['trans_cof'][third_mask,:] * \
                                    trans_color)
        
        # intersection = torch.zeros_like(rays_o)[:,0:2]
        # intersection[first_mask,:] = valid_intersection
        
        out_point = torch.zeros_like(rays_o)
        out_point[first_mask,:] = second_trans_ray['rays_o'][third_mask]
        
        out_dir = torch.zeros_like(rays_o)
        out_dir[first_mask,:] = second_trans_ray['rays_d'][third_mask]
        
        
        assert ((first_trans_ray['trans_cof'][second_mask,:]<1).all())
        assert ((second_trans_ray['trans_cof'][third_mask,:]<1).all())
        
        if self.reflection_flag:
            ref_render = self.renderer.render(ref_ray['rays_o'],ref_ray['rays_d'], cos_anneal_ratio=1.0)
            color_fine[first_mask,:] += ref_ray['ref_cof'][second_mask,:] * ref_render['color_fine'][second_mask,:]
        
        # an ugly way to transfer the mask to the original shape
        total_ref_mask_show = hit_mask.masked_scatter(hit_mask, totalRef_mask)
        valid_render_mask_show = total_ref_mask_show.masked_scatter(total_ref_mask_show,valid_render_mask)
        
        assert(total_ref_mask_show.shape[0] == color_fine.shape[0])
        assert(valid_render_mask_show.shape[0] == color_fine.shape[0])
        
        normal_all = torch.zeros_like(rays_o)
        normal_all[hit_mask] = normal
        
        output = {
            'color': color_fine,
            'gradient_error': (grad_error_1[second_mask] + grad_error_2[third_mask])/2,
            'mask': first_mask,
            'hit_mask': hit_mask,
            'total_ref_mask': total_ref_mask_show,
            'valid_render_mask': valid_render_mask_show,
            'normal': normal,
            'normal_all': normal_all,
            # 'intersection':intersection,
            'out_point': out_point,
            'out_dir': out_dir
        }
        
        return output 
    

    def render_with_transparent(self,rays_o, rays_d):
        """
        Render all pixel for visualization
        for transparent object, render it following the snell law
        for others, render it with scene appearance
        """
        trans_render_out = self.render_transparent(rays_o, rays_d)
        color_fine = trans_render_out['color']
        hit_mask = trans_render_out['hit_mask']
        simple_mask = ~hit_mask
        render_out = self.renderer.render(rays_o[simple_mask],rays_d[simple_mask])
        color_fine[simple_mask] = render_out['color']
        normal = torch.zeros_like(rays_o)
        normal[hit_mask] = trans_render_out['normal']
        out_dir = trans_render_out['out_dir']
        
        # n_samples = self.renderer.n_samples + self.renderer.n_importance
        # simple_normal = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
        # simple_normal = simple_normal.sum(dim=1)
        return {
            'color_fine': color_fine,
            'normals': normal,
            'hit_mask': hit_mask,
            'out_dir': out_dir
        }
    
    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0,min_z=None):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.tracer.sdf_network.sdf(pts),
                                min_z=min_z)