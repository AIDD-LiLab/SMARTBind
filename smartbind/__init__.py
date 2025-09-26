import logging
import sys

from smartbind.model.pl_train.contact import ContactPL
from smartbind.model.pl_train.binding import BindingPL
from smartbind.utils import load_smartbind_models

logger = logging.getLogger("SmartBind")

if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
