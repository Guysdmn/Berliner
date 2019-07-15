#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from data import core, log
import logging

""" solve function
Entry point to solution.
params: 
fin     = input .csv name leads file.
groupBy = split .csv to groups by groupBy factor. 'defult' for solve as one group.
mode    = distance matrix calculation factor, mode: 'driving','walking,'bicycling'.
disMat  = True: build new distance Matrixes, False: don't.
builder = distance matrix builder to use: 'google' / 'geo' / 'random'.
solver  = solver to use: 'or' / 'ortw' / 'aco' / 'cmp'. 'cmp': compare between all solvers.
plot    = True: plotting on map using google static map api key, False: don't.
"""
def solve(fin,groupBy='defult',mode='walking',disMat=True,builder='geo',solver='or',plot=False):
    # logger setup.
    logger = log.setup_logger('root')
    
    try:
        df = pd.read_csv(fin)
        df = df.dropna()
        df = df[1:50]
    except:
        logger.error("{} not found".format(fin))
        sys.exit()

    try:
        if(groupBy == 'defult'):
            core.defult_solver(df=df,mode=mode,disMat=disMat,builder=builder,solver=solver,plot=plot)
        else:
            core.group_solver(df=df,groupBy=groupBy,mode=mode,disMat=disMat,builder=builder,solver=solver,plot=plot)
    except:
        logger.critical("ERROR: Solve aborted")
        sys.exit()


if __name__ == "__main__":
    
    fin     = sys.argv[1] #'csv/*.csv'
    groupBy = 'defult'#'postal_code'
    solver  = 'or'
    builder = 'geo'
    disMat  = True
    mode    = 'walking'
    plot    = True 
    solve(fin=fin,groupBy=groupBy,mode=mode,disMat=disMat,builder=builder,solver=solver,plot=plot)
