import argparse
import datetime
import logging
import math
import random
import time
from os import path as osp
import os
import sys
import torch

# Add current directory to path for direct execution
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.backends.cudnn.benchmark = False
torch.set_num_threads(10)

from dataset import build_dataloader, build_dataset
from models import build_model
from utils import (MessageLogger, check_resume, get_root_logger, get_time_str, init_tb_logger,
                   make_exp_dirs, mkdir_and_rename, set_random_seed)
from utils.options import parse
from utils.misc import cp_options


def parse_options(root_path, is_train=True):
    """Parse options from command line and config file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train, debug=args.debug)

    # Single GPU settings
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed)

    return opt, args.opt


def init_loggers(opt):
    """Initialize loggers (TensorBoard only)."""
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='RCEstimate', log_level=logging.INFO, log_file=log_file)

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    """Create train and validation dataloaders."""
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(
                train_set,
                dataset_opt)

            num_iter_per_epoch = math.ceil(len(train_set) / dataset_opt['batch_size'])
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            opt['train']['total_iter'] = total_iters
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tBatch size: {dataset_opt["batch_size"]}'
                        f'\n\tIter per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif 'val' in phase:
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt)
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs, total_iters


def train_pipeline(root_path):
    """Main training pipeline."""
    start_time = time.time()

    # parse options
    opt, opt_path = parse_options(root_path, is_train=True)

    torch.backends.cudnn.benchmark = True

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # copy the option file to experiment folder
    cp_options(opt, opt_path)

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = build_model(opt)
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = build_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    start_time = time.time()
    device = torch.device('cuda' if opt['num_gpu'] > 0 else 'cpu')

    try:
        for epoch in range(start_epoch, total_epochs):
            iter_time = time.time()
            for train_data in train_loader:
                current_iter += 1

                # Move data to device
                for k, v in train_data.items():
                    if torch.is_tensor(v):
                        train_data[k] = v.to(device, non_blocking=True)

                train_data['iter'] = current_iter
                model.feed_data(train_data)
                model.optimize_parameters()
                iter_time = time.time() - iter_time

                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_time})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

                # validation
                if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

                # save models and training states
                if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter)

                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train']['warmup_iter'])
                iter_time = time.time()

            print('epoch:{} lr:{}'.format(epoch, model.get_current_learning_rate()[0]))

        # end of training
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f'End of training. Time consumed: {consumed_time}')

        if 'debug' in opt['name']:
            return

        logger.info('Save the latest model.')
        model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

        if opt.get('val') is not None and not 'test' in opt.get('name'):
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
        if tb_logger:
            tb_logger.close()

    except KeyboardInterrupt as e:
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f'Exception at iteration: {current_iter}, epoch:{epoch}. Time consumed: {consumed_time}')
        logger.info('Save the model.')
        model.save(epoch, current_iter)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
