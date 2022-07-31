import sys
from typing import Literal

from loguru import logger


Levels = Literal['TRACE', 'DEBUG', 'INFO']

# logger.info('Setup logger')
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level>: <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>")
logger.add(
    'logs.log', level='DEBUG', rotation='sunday',
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level>: <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>")
# logger.success('Setup logger completed!')
