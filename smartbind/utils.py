import logging
logger = logging.getLogger("SmartBind")

import os
from smartbind import BindingPL
import torch


def check_device(device):
    if 'cuda' in device:
        if torch.cuda.is_available():
            logger.info(f'{device} is available for training')
            return device
        else:
            logger.info(f'cuda is not available, switching to cpu')
            return 'cpu'
    elif 'cpu' in device:
        logger.info(f'{device} is available for training')
        return 'cpu'
    else:
        logger.info(f'Invalid device, switching to cpu')
        return 'cpu'


def load_smartbind_models(model_path, vs_mode, device='cpu', log=False):
    """
    Load SmartBind models
    """
    if model_path.endswith('.pth'):
        model = BindingPL(device=device, vs_mode=vs_mode)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=torch.device(device))
        pretrained_dict = {k: v
                           for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        if log:
            for k, v in pretrained_dict.items():
                print(f'Loading params {k} with shape {v.shape}')

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        not_loaded = [k for k in model_dict if k not in pretrained_dict]
        if not_loaded:
            print(f'Not loaded: {not_loaded}')

        return model

    model_list = []
    for file in os.listdir(model_path):
        if file.endswith('.pth'):
            model = BindingPL(device=device, vs_mode=vs_mode)
            model.to(device)
            model_dict = model.state_dict()
            pretrained_dict = torch.load(os.path.join(model_path, file), map_location=torch.device(device))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            if log:
                for k, v in pretrained_dict.items():
                    print(f'Loading params {k} with shape {v.shape}')

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.eval()
            model_list.append(model)

            not_loaded = [k for k in model_dict if k not in pretrained_dict]
            if not_loaded:
                print(f'Not loaded: {not_loaded}')

    return model_list


def load_single_smartbind_model(model_path, vs_mode, device='cpu'):
    """
    Load single SmartBind model
    """
    model = BindingPL(device=device, vs_mode=vs_mode)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=torch.device(device))
    pretrained_dict = {k: v
                       for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    for k, v in pretrained_dict.items():
        print(f'Loading params {k} with shape {v.shape}')

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    not_loaded = [k for k in model_dict if k not in pretrained_dict]
    if not_loaded:
        print(f'Not loaded: {not_loaded}')

    return model
