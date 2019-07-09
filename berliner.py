#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from data import core, log
import time
import logging

from mapsplotlib import mapsplot as mplt

""" solve function
Entry point to solution.
params: 
fin     = input .csv name leads file.
fout    = output .csv name for solutions.
groupBy = split .csv to groups by groupBy factor. 'defult' for solve as one group.
mode    = distance matrix calculation factor, mode: 'driving','walking,'bicycling'.
disMat  = True: build new distance Matrixes, False: don't.
builder = distance matrix builder to use: 'google' / 'geo' / 'random'.
solver  = solver to use: 'or' / 'ortw' / 'aco' / 'cmp'. 'cmp': compare between all solvers.
"""
def solve(fin,fout,groupBy='defult',mode='walking',disMat=True,builder='geo',solver='or'):
    # logger setup.
    logger = log.setup_logger('root')

    
    try:
        df = pd.read_csv(fin) #,usecols = ['name','latitude','longitude','postal_code'])
        df = df.dropna()
    except:
        logger.error("{} not found".format(fin))
        sys.exit()
    
    ### mplt test ###
    try:
        with open('data/api_key.txt', mode='r') as f:
            API_key = f.readline().strip()
            logger.info("Google API_KEY successfully connected")
    except:
        logger.error("Google API_KEY failed")
        raise
    mplt.register_api_key(API_key)
    # df = df.loc[np.r_[0:11930],:]
    # df['color'] = 'black'
    # df['size'] = 'medium'
    # df['value'] = 3
    # print(df)
    # mplt.heatmap(df['latitude'], df['longitude'], df['value'])
    # mplt.plot_markers(df)
    # mplt.density_plot(df['latitude'], df['longitude'])
    # mplt.polygons(df['latitude'], df['longitude'], df['postal_code'])

    try:
        if(groupBy == 'defult'):
            #
            # df = pd.read_csv(fin) #,usecols = ['name','latitude','longitude'])
            core.defult_solver(fin,fout,mode=mode,disMat=disMat,solver=solver)
        else:
            # df = pd.read_csv(fin) #,usecols = [groupBy,'name','latitude','longitude'])
            core.group_solver(df=df,groupBy=groupBy,mode=mode,disMat=disMat,builder=builder,solver=solver)
    except:
        logger.critical("ERROR: Solve aborted")
        sys.exit()


    
if __name__ == "__main__":
    
    fin     = sys.argv[1] #'csv/*.csv'
    fout    = sys.argv[2] #'csv/solution*.csv'
    groupBy = 'postal_code'
    solver  = 'or'
    builder = 'geo'
    disMat  = True
    mode    = 'walking'
    solve(fin=fin,fout=fout,groupBy=groupBy,mode=mode,disMat=disMat,builder=builder,solver=solver)
