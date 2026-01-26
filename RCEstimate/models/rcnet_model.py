from collections import OrderedDict
import logging
from models.base_model import BaseModel
from losses import build_loss
from utils.corres_utils import corres2color
from archs import build_network
from utils.logger import get_root_logger
import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
import os.path as osp
from matplotlib import cm

logger = logging.getLogger('RCEstimate')

def colormap(diff,thres):
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = cm.jet(diff_norm)[:,:, :3]
    return diff_cm[:,:,::-1]

def EPE_error_withbatch(pred, gt, mask=None):
    """
    Args:
        pred: [B,2,H,W] tensor
        gt: [B,2,H,W] tensor
        mask: [B,1,H,W] tensor
    """
    assert pred.shape == gt.shape
    if mask is not None:
        assert mask.shape[0] == pred.shape[0]
        mask = mask.bool()
    else:
        mask = torch.ones_like(pred).bool()
    batch_average_error = []
    for i in range(pred.shape[0]):
        pred_i = pred[i:i+1]
        gt_i = gt[i:i+1]
        mask_i = mask[i:i+1]
        error = torch.norm(pred_i-gt_i, dim=1, p=2,keepdim=True)
        error = error[mask_i]
        batch_average_error.append(error.mean().item())
    
    return np.mean(np.array(batch_average_error))

def px_error_withbatch(pred, gt, mask=None, threshold=0.05):
    """
    Args:
        pred: [B,2,H,W] tensor
        gt: [B,2,H,W] tensor
        mask: [B,1,H,W] tensor
    """
    assert pred.shape == gt.shape
    if mask is not None:
        assert mask.shape[0] == pred.shape[0]
        mask = mask.bool()
    else:
        mask = torch.ones_like(pred).bool()
    batch_average_error = []
    for i in range(pred.shape[0]):
        pred_i = pred[i:i+1]
        gt_i = gt[i:i+1]
        mask_i = mask[i:i+1]
        error = torch.norm(pred_i-gt_i, dim=1, p=2,keepdim=True)
        error = error[mask_i]
        batch_average_error.append((error > threshold).float().mean().item())
    
    return np.mean(np.array(batch_average_error))

def recon_refract_img(background:torch.Tensor,correspondence:torch.Tensor,valid_mask:torch.Tensor):

    def check(h,w,tensor):
        assert tensor.dim() == 4
        assert tensor.shape[-2] == h
        assert tensor.shape[-1] == w
    """perform image reconstruction
    """
    # I = _reflection + Warp(C, B)
    b,c,h, w = background.shape

    if correspondence.dim() == 3:
        correspondence = correspondence.unsqueeze(0)
    check(h,w,correspondence)
    check(h,w,valid_mask)

    #* grid_sample range from -1 to 1
    correspondence = correspondence.permute(0,2,3,1)
    correspondence = correspondence*2-1
    refract_img = torch.nn.functional.grid_sample(background, correspondence, mode='bilinear', padding_mode='border',align_corners=True)
    img = torch.zeros_like(background)
    img[valid_mask.expand(b,3,h,w)] = refract_img[valid_mask.expand(b,3,h,w)]

    return img


# This implementation is based on BasicSR
# Refractive Correspondence Network
class RCNet(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.visSavePath = opt['val'].get('savePath', None)
        self.initNetworks()
        self.cri_corres = None  
        if self.is_train:
            self.initLoss()  
            self.network.train()
            self.setup_optimizers()
            self.setup_schedulers()
    
    def buildNet(self, net_opt, path_key):
        net = build_network(net_opt)
        net = self.model_to_device(net)

        load_path = self.opt['path'].get('pretrain_' + path_key, None)
        # load pretrained models
        if load_path is not None:
            params = 'params'
            self.load_network(net, load_path, self.opt['path'].get('strict_load_'+path_key, True), params)
        return net
    
    def initNetworks(self):
        self.network = self.buildNet(self.opt['network'], 'network')

    def setup_optimizers(self):
        optim_params = [param for param in self.network.parameters() if param.requires_grad]
        optim_type = self.opt['train']['optim_net'].pop('type')
        self.optimizer_net = self.get_optimizer(optim_type, optim_params, **self.opt['train']['optim_net'])
        self.optimizers.append(self.optimizer_net)
    
    def initLoss(self):
        train_opt = self.opt['train']
        self.cri_corres = build_loss(train_opt['corres_opt']).to(self.device)
    
    def map_loss(self, output, loss_dict, corres_gamma=0.9):
        gt_correspondence = self.gt['correspondence']
        corres_loss = 0
        n_predictions = len(output['correspondence'])
        for idx,pred_corres in enumerate(output['correspondence']):
            weight = corres_gamma**(n_predictions-1-idx)
            loss = self.cri_corres(pred_corres, gt_correspondence, self.gt['valid_mask'])
            corres_loss += weight*loss
        loss_dict['l_corres'] = corres_loss
        loss = corres_loss
        return loss
    
    def computeLoss(self):
        loss_dict = OrderedDict()
        loss = self.map_loss(self.output, loss_dict)
        return loss_dict, loss,

    def optimize_parameters(self):
        self.output = self.network(self.inputs)
        loss_dict, l_total = self.computeLoss()
        
        self.optimizer_net.zero_grad()
        l_total.backward()
        self.optimizers[0].step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test_forward(self):
        self.network.eval()
        with torch.no_grad():
            self.output = self.network(self.inputs)
        self.network.train()
    
    def _compute_metric(self, metric_name, pred_corres, gt_corres, valid_mask):
        """Compute single metric based on metric name."""
        if metric_name == 'epe':
            return EPE_error_withbatch(pred_corres, gt_corres, valid_mask)
        elif metric_name == 'px3':
            return px_error_withbatch(pred_corres, gt_corres, valid_mask, threshold=0.03)
        elif metric_name == 'px5':
            return px_error_withbatch(pred_corres, gt_corres, valid_mask, threshold=0.05)
        elif metric_name == 'px10':
            return px_error_withbatch(pred_corres, gt_corres, valid_mask, threshold=0.1)
        else:
            raise ValueError(f'Unsupported metric: {metric_name}')

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        dataset_name = dataloader.dataset.opt['name']
        metrics_list = self.opt['val'].get('metrics', None)
        with_metrics = metrics_list is not None

        if with_metrics:
            self.metric_results = {metric: 0 for metric in metrics_list}

        if self.opt.get('pbar', True):
            pbar = tqdm(total=len(dataloader), unit='image')

        with torch.no_grad():
            for idx, val_data in enumerate(dataloader):
                self.feed_data(val_data)
                self.test_forward()

                if self.opt['val'].get('save_img', False):
                    save_path = osp.join(self.opt['path']['visualization'], dataset_name, 'iter_{}'.format(current_iter))
                    os.makedirs(save_path, exist_ok=True)
                    visual_path = osp.join(save_path, f'correspondence--{idx}.png')
                    self.save_visuals(visual_path, self.output, self.gt)

                if with_metrics:
                    pred_corres = self.output['correspondence'][-1]
                    gt_corres = self.gt['correspondence']
                    valid_mask = self.gt['valid_mask']

                    for metric_name in metrics_list:
                        metric_value = self._compute_metric(metric_name, pred_corres, gt_corres, valid_mask)
                        self.metric_results[metric_name] += metric_value

                torch.cuda.empty_cache()
                if self.opt.get('pbar', True):
                    pbar.update(1)
                    pbar.set_description(f'Testing')

            if self.opt.get('pbar', True):
                pbar.close()

            if with_metrics:
                for metric in self.metric_results.keys():
                    self.metric_results[metric] /= (idx + 1)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    # We use the flo file of the optical flow to save correspondence.    
    def write_flo_file(self,flow, filename):
        H, W, _ = flow.shape
        with open(filename, 'wb') as f:
            np.array([202021.25], dtype=np.float32).tofile(f)
            np.array([W, H], dtype=np.int32).tofile(f)
            flow.astype(np.float32).tofile(f)
    
    def save_single_result(self,output_folder,basename):
        pred_corres = self.output['correspondence'][-1].permute(0,2,3,1).detach().cpu().numpy()
        corres_path = osp.join(output_folder, f'{basename}_pred_corres.flo')
        self.write_flo_file(pred_corres[0], corres_path)
    
    def validation_for_recon(self,dataloader,output_folder):       
        os.makedirs(output_folder,exist_ok=True)
        with torch.no_grad():
            for idx, val_data in enumerate(dataloader):
                self.feed_data(val_data)
                self.test_forward()
                self.save_single_result(output_folder, str(idx).zfill(3))
                visual_path=osp.join(output_folder, f'visualization--{str(idx).zfill(3)}.png')
                self.save_visuals(visual_path, self.output, gt=None)                                   
                torch.cuda.empty_cache()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name};\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\t'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def save_visuals(self, path, pred, gt=None):
        gt_input_img = 255*self.gt['input_img'].permute(0,2,3,1).detach().cpu().numpy()
        background = 255*self.gt['background'].permute(0,2,3,1).detach().cpu().numpy()
        gt_valid_mask = 255*torch.tile(self.gt['valid_mask'],[1,3,1,1]).permute(0,2,3,1).detach().cpu().numpy()
        # input_map = np.concatenate([gt_input_img, background, gt_valid_mask], axis=2)
        input_map = np.concatenate([gt_input_img], axis=2)
        pred_corres = pred['correspondence'][-1].permute(0,2,3,1).detach().cpu().numpy()
        pred_corres_map = corres2color(pred_corres, mask=gt_valid_mask[:,:,:,:1]/255)
        
        if gt:
            gt_corres = gt['correspondence'].permute(0,2,3,1).detach().cpu().numpy()
            gt_corres_map = corres2color(gt_corres, mask=gt_valid_mask[:,:,:,:1]/255)
            error_map_list = []
            for idx in range(gt_corres.shape[0]):
                error = np.linalg.norm(pred_corres[idx]-gt_corres[idx], axis=2)
                error_map = colormap(error, 0.1)[:,:,::-1]
                m = gt_valid_mask[idx]
                error_map[m==0] = 0
                error_map_list.append(error_map)
            error_map = np.stack(error_map_list,axis=0)
            refract_img = recon_refract_img(self.gt['background'],pred['correspondence'][-1],self.gt['valid_mask'].bool())
            refract_img = 255*refract_img.permute(0,2,3,1).detach().cpu().numpy()
            
            output_map = np.concatenate([input_map, gt_corres_map,pred_corres_map,255*error_map,refract_img], axis=2)
        else:
            refract_img = recon_refract_img(self.gt['background'],pred['correspondence'][-1],self.gt['valid_mask'].bool())    
            refract_img = 255*refract_img.permute(0,2,3,1).detach().cpu().numpy()
            color_error = np.abs(input_map - refract_img)
            m = gt_valid_mask[0:1]
            color_error[m==0] = 0
            output_map = np.concatenate([refract_img,input_map, color_error,pred_corres_map], axis=2)
        
        output_map = output_map.reshape(output_map.shape[0]*output_map.shape[1],output_map.shape[2],output_map.shape[3])
        output_map = output_map.clip(0,255).astype(np.uint8)
        
        cv2.imwrite(path, output_map[:,:,::-1])

    def feed_data(self,data):
        self.gt = OrderedDict()
        for key in ['input_img','background','correspondence','valid_mask']:
            self.gt[key] = data[key].to(self.device) if key in data.keys() else None
        
        self.inputs = {
            'input_img':self.gt['input_img'],
            'valid_mask':self.gt['valid_mask'],
            'background':self.gt['background'],
        }
    
    def update_learning_rate(self,current_iter, warmup_iter=-1):          
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)
    
    def save(self, epoch, current_iter):
        self.save_network(self.network, 'network', current_iter)
        self.save_training_state(epoch, current_iter)
        
