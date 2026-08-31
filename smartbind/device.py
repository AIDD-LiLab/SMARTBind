import logging

import torch

logger = logging.getLogger("SmartBind")


def resolve_device(device=None):
    """Normalise a user-supplied device string, falling back to CPU when needed.
    
    Returns a string that can be handed to ``torch.device``.
    """
    requested = 'auto' if device is None else str(device).strip().lower()

    if requested in ('auto', ''):
        resolved = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'device=auto resolved to {resolved}')
        return resolved

    if requested == 'cpu':
        return 'cpu'

    if requested == 'gpu':
        requested = 'cuda'

    if requested == 'cuda' or requested.startswith('cuda:'):
        if not torch.cuda.is_available():
            logger.warning(
                f"device='{device}' was requested but no CUDA device is visible to "
                f"torch; falling back to cpu")
            return 'cpu'
        if requested.startswith('cuda:'):
            index = requested.split(':', 1)[1]
            if not index.isdigit():
                logger.warning(f"device='{device}' is not a valid CUDA index; using cuda:0")
                return 'cuda:0'
            if int(index) >= torch.cuda.device_count():
                logger.warning(
                    f"device='{device}' was requested but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are visible; using cuda:0")
                return 'cuda:0'
        return requested

    logger.warning(f"device='{device}' is not recognised; falling back to cpu")
    return 'cpu'


# Kept as the historical name used by the training entry points.
check_device = resolve_device


def lightning_accelerator(device):
    """Translate a resolved device string into Lightning ``Trainer`` arguments.

    Returned as ``(accelerator, devices)``.  Passing these explicitly matters: the
    Trainer default is ``accelerator='auto'``, which would put training on a GPU even
    when the caller asked for CPU, leaving the model and the trainer on different
    devices.
    """
    device = resolve_device(device)
    if device == 'cpu':
        return 'cpu', 1
    if ':' in device:
        return 'gpu', [int(device.split(':', 1)[1])]
    return 'gpu', 1
