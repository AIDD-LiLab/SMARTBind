"""Seeding that does not leak into the caller's random number generators.

`BindingPL.__init__` and `ContactPL.__init__` seed the global RNGs so a model built for
training gets a reproducible weight initialisation.  That is right when the model keeps
those weights.  On the inference path they are replaced immediately by `load_state_dict`,
so the seeding achieves nothing except resetting `random`, NumPy and torch out from under
whoever called us -- ten times over when the ten-fold ensemble is loaded, which is also
where the ten `Seed set to 42` lines come from.

Use as a pair around the initialisation:

    token = seed_model_init(seed, isolate=..., verbose=...)
    ...  # build the layers
    restore_rng(token)

With ``isolate=False`` (the default, and what both training entry points get) this is
exactly `seed_everything(seed)` and `restore_rng` does nothing, so training behaviour is
unchanged.  With ``isolate=True`` the caller's generator state is put back afterwards.
"""
import contextlib
import inspect
import logging
import random

import numpy as np
import torch
from pytorch_lightning import seed_everything

# pytorch_lightning grew a `verbose` flag on seed_everything in 2.2; on 2.0/2.1 the only
# way to keep it quiet is to mute the logger it writes through.
_SEED_TAKES_VERBOSE = "verbose" in inspect.signature(seed_everything).parameters
_SEED_LOGGER = "lightning_fabric.utilities.seed"


def _seed_quietly(seed, verbose):
    if _SEED_TAKES_VERBOSE:
        seed_everything(seed, verbose=verbose)
        return
    if verbose:
        seed_everything(seed)
        return
    log = logging.getLogger(_SEED_LOGGER)
    previous = log.disabled
    log.disabled = True
    try:
        seed_everything(seed)
    finally:
        log.disabled = previous


def seed_model_init(seed, isolate=False, verbose=True):
    """Seed the global RNGs for weight initialisation; returns a token for restore_rng."""
    token = None
    if isolate:
        token = (random.getstate(),
                 np.random.get_state(),
                 torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    _seed_quietly(seed, verbose)
    return token


def restore_rng(token):
    """Put back the generator state captured by seed_model_init(isolate=True)."""
    if token is None:
        return
    py_state, np_state, torch_state, cuda_states = token
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
