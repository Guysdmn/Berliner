#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import googlemaps
from geopy import distance
import time
from itertools import islice, tee
import logging

logger = logging.getLogger('root')

from solver import orsolver, acosolver, orsolvertw
from mapsplotlib import mapsplot as mplt

"""solve function

params: 
disMatName = 
solver     =
"""
def explore(disMatName,solver='or'):
    # initialize solver by solver arg.
    if(solver=='or'):
        #
        OR_solver = orsolver.OR_solver(distance_matrix=disMatName)
        #
        start = time.time()
        perm  = OR_solver.run()   # OR_tools
    elif(solver=='ortw'):
        #
        OR_solvertw = orsolvertw.OR_solver_TW(fin=fout)
        #
        start = time.time()
        perm  = OR_solvertw.run()   # OR_tools time windows
    elif (solver=='aco'):
        #
        disf = pd.DataFrame(pd.read_csv(disMatName))
        disf = disf.drop(disf.columns[[0]], axis=1)
        fd   = disf.values.copy()
        #
        ACO_solver  = acosolver.ACO_solver(Graph=fd,seed=345)
        #
        start = time.time()
        perm  = ACO_solver.run() # ACO
    else:
        raise
    
    return start, perm

"""group_solver function

params: 
df      = 
groupBy = 
mode    = 
disMat  = 
builder = 
solver  =
"""
def group_solver(df,groupBy,mode,disMat,builder,solver):
    try:
        try:
            with open('data/api_key.txt', mode='r') as f:
                API_key = f.readline().strip()
                mplt.register_api_key(API_key)
                logger.info("Google static map API_KEY successfully registered")
        except:
            logger.error("Google API_KEY acquired failed")
            raise
        logger.info("-------------------Prepering dataframe for solution-------------------")
        # cast groupBy column to int64 if groupBy is numeric
        if np.issubdtype(df[groupBy].dtype, np.number):
            df[groupBy] = df[groupBy].astype(np.int64)
            logger.info("Clustering by numeric values")
        
        # split data by groupBy into dictionary, key = groupBy column.
        # df = df.sort_values(by=[groupBy], ascending=True, na_position='first').dropna() # maybe unnessesery
        uniqueNames = df[groupBy].unique()
        logger.info("Total of {} different {}".format(len(uniqueNames),groupBy))
        dfDict = {value : pd.DataFrame for value in uniqueNames}
        for key in dfDict.keys():
            dfDict[key] = df[:][df[groupBy] == key]

        # build distance matrix for each group.
        # solve tsp for each group.
        for group, data in dfDict.items():
            logger.info('----------------------STARTING {} {}----------------------'.format(groupBy,group))
            distance_mat = 'solution/{}_distance.csv'.format(group)
            n = data.shape[0]
            solution = (list(range(n)), 0)

            # build distance matrix for group if disMat=True
            if disMat:
                if builder == 'google':
                    google_distance_builder(data=data,fout=distance_mat,mode=mode)
                elif builder == 'geo':
                    geo_distance_builder(data=data,fout=distance_mat)
                else: # random builder
                    random_distance_builder(fout=distance_mat,nxn=n,min_value=5,max_value=100)

            # find solution only for groups bigger then 2
            if(n > 2):
                logger.info("Searching for optimal route...")
                start,solution = explore(distance_mat,solver)
                end = time.time()

                logger.info("{}.csv number of leads in {}: {}".format(group,groupBy,n))
                logger.info("Order of {}:{}".format(mode,solution[0]))
                logger.info("Objective: {}".format(solution[1]))
                logger.info("{} solver timer: {:.2f} minutes".format(solver.upper(),(end-start)/60))
            else:
                logger.info("Group contains less then three points...")

            fname = 'solution/{}_solution.csv'.format(group)
            csv_export(data=data,order=solution[0],fout=fname)
            pname = 'solution/{}.png'.format(group)
            heatmap_export(data=data,fout=pname)

        return start,solution
    except:
        logger.error("Group solver failed")
        raise


"""defult_solver function

params: 
fin  = 
fout = output .csv distance matrix file.
mode = 
"""
def defult_solver(fin,fout,mode='walking'):
    distance_mat = 'solution/distance_matrix.csv'

    # build distance matrix
    try:
        #google_distance_builder(data=data,fout=distance_mat,mode=mode)
        geo_distance_builder(data=data,fout=distance_mat)
        # random_distance_builder(fout=distance_mat,nxn=n,min_value=5,max_value=100)
    except:
        raise
    
    try:
        logger.info('-----------------------START----------------------------')
        disf = pd.DataFrame(pd.read_csv(distance_mat))
        disf = disf.drop(disf.columns[[0]], axis=1)
        fd   = disf.values.copy()
        
        # initialize solvers
        # OR_solver   = orsolver.OR_solver(distance_matrix=distance_mat)
        # #OR_solvertw = orsolvertw.OR_solver_TW(fin=fout)
        # ACO_solver  = acosolver.ACO_solver(Graph=fd,seed=345)

        # # solve
        # start = time.time()
        # or_perm = OR_solver.run()   # OR_tools
        # end = time.time()
        # print("OR solver timer: {:.2f}".format((end-start)/60), "minutes")
        # print('---------------------------------------------------------')
        # start = time.time()
        # aco_perm = ACO_solver.run() # ACO
        # end = time.time()
        # print("ACO solver timer: {:.2f}".format((end-start)/60), "minutes")

        # print('-----------------------FINNISH---------------------------')
    except:
        raise
    try:
        core.csv_export(fin=fin,order=or_perm,fout=fout)
    except:
        raise


""" google_distance_builder function
building distance matrix using google distance matrix API, save matrix as fout.csv
builder function should be call only ones for *.csv file (Google API $$$...)
*** Google API key requiered ***
params: 
data = input data frame.
fout = output .csv distance matrix file.
"""
def google_distance_builder(data,fout,mode):

    logger.critical("Using google key for distance matrix")
    # Google Maps API web service.
    try:
        with open('data/api_key.txt', mode='r') as f:
            API_key = f.readline().strip()
            # Connect to googlemaps.
            Gmaps = googlemaps.Client(key=API_key)
            logger.info("Google distance matrix API_KEY successfully registered")
    except:
        logger.error("Google API_KEY failed")
        raise

    try:
        # Init distance matrix - will be used to store calculated distances and times.
        disMat   = pd.DataFrame(0,columns=data.name.unique(), index=data.name.unique())
        apiCalls = 0

        # Start building timer.
        start = time.time()

        # Loop through each row in the data frame.
        for (i1, row1) in data.iterrows():
            # Assign latitude and longitude as origin points.
            LatOrigin  = row1['latitude']
            LongOrigin = row1['longitude']
            origin     = (LatOrigin,LongOrigin)

            # Loop through unvisited paths in the data frame (decrease API calls $$$).
            for (i2, row2) in islice(data.iterrows(),i1):
                # Assign latitude and longitude as destination points.
                LatDest     = row2['latitude']
                LongDest    = row2['longitude']
                destination = (LatDest,LongDest)

                # Skip condition, matrix diagonal.
                if(origin == destination):
                    continue
                
                # Check geo distance, if greater then maxDistance append max distance and skip.
                maxDistance = 3500
                if(distance.distance(origin, destination).m > maxDistance):
                    disMat[row1['name']][row2['name']] = 10^4
                    disMat[row2['name']][row1['name']] = 10^4
                    continue
                
                # Pass origin and destination variables to distance_matrix googlemaps function.
                result = Gmaps.distance_matrix(origin, destination, mode=mode)
                apiCalls += 1

                # Create resault distance(meters), duration(minuts).
                dis = int(result['rows'][0]['elements'][0]['distance']['value'])
                dur = [int(s) for s in result['rows'][0]['elements'][0]['duration']['text'].split() if s.isdigit()][0]

                # Assert values to distance mat, both ways (by distance(meters) or by duration(minuts)).
                disMat[row1['name']][row2['name']] = dur
                disMat[row2['name']][row1['name']] = dur

        # Stop building timer
        end = time.time()
        
        # Save as .csv file
        disMat.to_csv(fout)

        # Print stats
        logger.info("-----------------------------------------------------------------------")
        logger.info("Built distane matrix in: {:.2f} minutes with {} Google API calls".format((end-start)/60,apiCalls))
        logger.info("Distance saved to: {}".format(fout))
        logger.info("-----------------------------------------------------------------------")
    except:
        logger.error("Google distance matrix failed")
        raise


""" geo_distance_builder for dev mode
build  distance matrix using geopy distance calculator, save matrix as fout.csv
distance calculator 
              model             major (km)   minor (km)     flattening
ELLIPSOIDS = {'WGS-84':        (6378.137,    6356.7523142,  1 / 298.257223563),
              'GRS-80':        (6378.137,    6356.7523141,  1 / 298.257222101),
              'Airy (1830)':   (6377.563396, 6356.256909,   1 / 299.3249646),
              'Intl 1924':     (6378.388,    6356.911946,   1 / 297.0),
              'Clarke (1880)': (6378.249145, 6356.51486955, 1 / 293.465),
              'GRS-67':        (6378.1600,   6356.774719,   1 / 298.25),}
WGS-84 ellipsoid model by default, which is the most globally accurate.
see : https://geopy.readthedocs.io/en/stable/#module-geopy.distance
params:
data = input data frame.
fout = output .csv distance matrix file.
"""
def geo_distance_builder(data,fout):
    try: 
        logger.info("Building geographical distance matrix...")   
        # Init distance matrix - will be used to store calculated distances.
        disMat = pd.DataFrame(0,columns=data.name.unique(), index=data.name.unique())

        # Start building timer.
        start = time.time()

        # Loop through each row in the data frame.
        for (i1, row1) in data.iterrows():
            # Assign latitude and longitude as origin points.
            LatOrigin  = row1['latitude']
            LongOrigin = row1['longitude']
            origin     = (LatOrigin,LongOrigin)

            # Loop through unvisited paths in the data frame (decrease API calls $$$).
            for (i2, row2) in islice(data.iterrows(),i1):
                # Assign latitude and longitude as destination points.
                LatDest     = row2['latitude']
                LongDest    = row2['longitude']
                destination = (LatDest,LongDest)

                # Skip condition, matrix diagonal.
                if(origin == destination):
                    continue
                
                # Get geo distance
                value = distance.distance(origin, destination).m
                # logger.info(value)
                maxDistance = 3500
                if(value > maxDistance):
                    disMat[row1['name']][row2['name']] = 10^4
                    disMat[row2['name']][row1['name']] = 10^4
                    continue

                disMat[row1['name']][row2['name']] = value
                disMat[row2['name']][row1['name']] = value

        # Stop building timer
        end = time.time()

        # Save as .csv file
        disMat.to_csv(fout)

        # Print stats
        logger.info("Built distane matrix in: {:.2f} minutes with geo_distance_builder".format((end-start)/60))
        logger.info("Distance saved to: {}".format(fout))
    except:
        logger.error("Geo distance matrix failed")
        raise


""" random_distance_builder
build random distance matrix size n X n with random leads name.
params:
fout      = output .csv distance matrix file.
nxn       = size of matrix.
min_value = minimum distance/duration value between leads.
max_value = maximum distance/duration value between leads.
"""
def pd_fill_diagonal(df_matrix, value=0): 
    mat = df_matrix.values
    n = mat.shape[0]
    mat[range(n), range(n)] = value
    return pd.DataFrame(mat)

def random_distance_builder(fout, nxn, min_value=1, max_value=100):
    try:
        # Create leads name.
        names = ['s' + str(x) for x in range(1,nxn+1)]
        # Create random value distance matrix.
        df = pd.DataFrame(np.random.randint(min_value,max_value,size=(nxn, nxn)),columns=names,index=names)
        # Fill diagonal with 0 value.
        pd_fill_diagonal(df,0)
        # Check "geo distance" like.
        df = df.apply(lambda x: [y if y <= 80 else (max_value*max_value) for y in x])
        # Save to fout.csv file.
        df.to_csv(fout)
    except:
        logger.error("Random distance matrix failed")
        raise

""" csv_export function
gets leads data from fin.csv, puts in fout.csv in order.
params:
fin   = input .csv file name, unordered. 
order = new order permutation.
fout  = output .csv file name, ordered by order arg. 
"""
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def csv_export(data,order,fout):
    try:
        # Create new data frame.
        orderd = pd.DataFrame(columns=list(data.columns))

        # Append to new data frame in order.
        for idx in order:
            orderd = orderd.append(data.iloc[idx])

        # Save ordered to *.csv file.
        orderd.to_csv(fout, index=False)
        logger.info("Solution saved to: {}".format(fout))
    except:
        logger.error("csv export failed")
        raise


"""

"""
def heatmap_export(data,fout):
    try:
        path = '/Users/Guy/Desktop/repos/Berliner/'
        mplt.density_plot(data['latitude'], data['longitude'],toFile=os.path.join(path,fout))
        logger.info("Heatmap saved to: {}".format(fout))
    except:
        logger.error("print heatmap failed")