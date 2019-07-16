#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from data import core, log
import logging

def solve(fin,groupBy="",mode="walking",disMat=True,builder="geo",solver="or",plot=False):
    """ solve function
    Entry point to solution.
    
    :param string fin     : input .csv leads file name
    :param string groupBy : split .csv to groups by groupBy factor. 'defult' for solve as one group
    :param string mode    : distance matrix calculation factor, mode: 'driving','walking,'bicycling'
    :param boolean disMat : True: build new distance Matrixes, False: don't
    :param string builder : distance matrix builder to use: 'google' / 'geo' / 'random'
    :param string solver  : solver to use: 'or' / 'ortw' / 'aco' / 'cmp'. 'cmp': compare between all solvers
    :param boolean plot   : True: plotting on map using google static map api key, False: don't
    
    :return: None
    """
    # logger setup.
    console = True
    logger = log.setup_logger('root',console=console)
    
    try:
        df = pd.read_csv(fin)
        df = df.dropna()
        df = df[1:500]
    except:
        logger.error("{} not found".format(fin))
        sys.exit()

    ### add 'cmp' handler ###

    try:
        if not groupBy:
            core.defult_solver(df=df,mode=mode,disMat=disMat,builder=builder,solver=solver,plot=plot)
        else:
            core.group_solver(df=df,groupBy=groupBy,mode=mode,disMat=disMat,builder=builder,solver=solver,plot=plot)
    except:
        logger.critical("ERROR: Solve aborted")
        sys.exit()


if __name__ == "__main__":
    
    fin     = sys.argv[1] #'csv/*.csv'
    groupBy = 'postal_code'
    solver  = 'aco'
    builder = 'geo'
    disMat  = True
    mode    = 'walking'
    plot    = True

    solve(fin=fin,groupBy=groupBy,mode=mode,disMat=disMat,builder=builder,solver=solver,plot=plot)
