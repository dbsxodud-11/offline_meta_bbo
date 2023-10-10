import logging

def get_logger(filename, mode='w'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    return logger
