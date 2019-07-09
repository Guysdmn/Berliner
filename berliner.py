#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from data import core, log
import time
import logging

# from mapsplotlib import mapsplot as mplt

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
    # try:
    #     with open('data/api_key.txt', mode='r') as f:
    #         API_key = f.readline().strip()
    #         mplt.register_api_key(API_key)
    #         logger.info("Google API_KEY successfully registered")
    # except:
    #     logger.error("Google API_KEY acquired failed")
    #     raise

    # df['color'] = 'black'
    # df['size'] = 'medium'
    # df['value'] = 3
    # path = '/Users/Guy/Desktop/repos/Berliner/solution'
    # mplt.heatmap(df['latitude'], df['longitude'], df['value'],toFile=os.path.join(path,'1'))
    # mplt.plot_markers(df,toFile=os.path.join(path,'2.jpeg'))
    # mplt.density_plot(df['latitude'], df['longitude'],toFile=os.path.join(path,'3.jpeg'))
    # mplt.polygons(df['latitude'], df['longitude'], df['postal_code'],toFile=os.path.join(path,'4.jpeg'))
    # mplt.scatter(df['latitude'], df['longitude'], df['postal_code'],toFile=os.path.join(path,'5.jpeg'))
    # mplt.polyline(df['latitude'], df['longitude'], closed=True,toFile=os.path.join(path,'6.jpeg'))

    try:
        if(groupBy == 'defult'):
            #
            core.defult_solver(fin,fout,mode=mode,disMat=disMat,solver=solver)
        else:
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
