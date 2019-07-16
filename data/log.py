#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import logging.handlers

def setup_logger(name,console):
    """ setup_logger function

    :param string name     : name of logger to be referenced from any file
    :param boolean console : print log to console or not
    """
    # logger settings
    log_file = "log.txt"
    log_format = "[%(levelname)s] >> %(message)s"
    log_filemode = "w" # w: overwrite; a: append

    # setup logger
    logging.basicConfig(filename=log_file, format=log_format, filemode=log_filemode ,level=logging.DEBUG)
    logger = logging.getLogger(name)

    # print log messages to console
    if console:
        consoleHandler = logging.StreamHandler()
        logFormatter = logging.Formatter(log_format)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    return logger