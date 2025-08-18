import os
import logging
from pie_toolbox.core.common.constants import ROOT_DIR

LOG_DIR = os.path.join(ROOT_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)


def setup_root_logger(name: str, log_file_name: str = None,
                      verbose_console: bool = True):
    """
    Настраивает root логгер (один раз), от которого будут наследоваться остальные.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # Уже настроен

    if not log_file_name:
        log_file_name = f'log_{name}.txt'

    log_path = os.path.join(LOG_DIR, log_file_name)

    # File handler
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose_console else logging.WARNING)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Получить логгер по имени (дочерний логгер от root).
    """
    return logging.getLogger(name)


def section(title: str, logger: logging.Logger, level: int = logging.INFO):
    """
    Prints a section title with decorative bars to the specified logger at the given level.

    Parameters
    ----------
    title : str
        The title of the section.
    logger : logging.Logger
        The logger instance to use.
    level : int, optional
        The logging level to use for the section title (e.g., logging.INFO, logging.DEBUG).
        Defaults to logging.INFO.
    """
    bar = '=' * 60

    logger.log(level, f'{bar}')
    logger.log(level, f'{title.center(60)}')
    logger.log(level, f'{bar}')
