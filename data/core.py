#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
# import sys
import pathlib
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


def explore(disMatName,solver='or'):
    """ solve function

    params: 
    disMatName = 
    solver     =
    """
    # initialize solver by solver arg.
    start = time.time()
    if(solver=='or'):
        #
        OR_solver = orsolver.OR_solver(distance_matrix=disMatName)
        #
        perm  = OR_solver.run()   # OR_tools
    elif(solver=='ortw'):
        #
        OR_solvertw = orsolvertw.OR_solver_TW(fin=fout)
        #
        perm  = OR_solvertw.run()   # OR_tools time windows
    elif (solver=='aco'):
        #
        disf = pd.DataFrame(pd.read_csv(disMatName))
        disf = disf.drop(disf.columns[[0]], axis=1)
        fd   = disf.values.copy()
        #
        ACO_solver  = acosolver.ACO_solver(Graph=fd,seed=345)
        #
        perm  = ACO_solver.run() # ACO
    else:
        raise
    
    return start, perm


def group_solver(df,groupBy,mode,disMat,builder,solver,plot):
    """ group_solver function

    params: 
    df      = 
    groupBy = 
    mode    = 
    disMat  = 
    builder = 
    solver  =
    plot    =
    """
    try:
        if plot is True:
            mapsplot_register()

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
            group_size   = data.shape[0]
            solution     = (list(range(group_size)), 0)

            # build distance matrix for group if disMat=True
            if disMat is True:
                if builder == 'google':
                    google_distance_builder(data=data,fout=distance_mat,mode=mode)
                elif builder == 'geo':
                    geo_distance_builder(data=data,fout=distance_mat)
                else: # random builder
                    random_distance_builder(fout=distance_mat,nxn=group_size,min_value=5,max_value=100)

            # find solution only for groups bigger then 2
            if(group_size > 2):
                logger.info("Searching for optimal route...")
                logger.info("{}.csv number of leads in {}: {}".format(group,groupBy,group_size))

                start,solution = explore(distance_mat,solver)
                end = time.time()

                logger.info("Order of {}:{}".format(mode,solution[0]))
                logger.info("Objective: {}".format(solution[1]))
                logger.info("{} solver timer: {:.2f} minutes".format(solver.upper(),(end-start)/60))
            else:
                logger.info("Group contains less then three points...")

            fname = 'solution/{}_solution.csv'.format(group)
            csv_export(data=data,order=solution[0],fout=fname)
            if plot is True:
                mapsplot_export(data=data,fout=group,groupBy=groupBy)

        return start,solution
    except:
        logger.error("Group solver failed")
        raise


def defult_solver(df,mode,disMat,builder,solver,plot):
    """ defult_solver function

    params: 
    fin  = 
    mode = 
    """
    if plot is True:
        mapsplot_register()

    try:
        logger.info('----------------------STARTING----------------------')
        # build distance matrix
        # build distance matrix for group if disMat=True
        if disMat is True:
            distance_mat = 'solution/distance_matrix.csv'
            if builder == 'google':
                google_distance_builder(data=df,fout=distance_mat,mode=mode)
            elif builder == 'geo':
                geo_distance_builder(data=df,fout=distance_mat)
            else: # random builder
                random_distance_builder(fout=distance_mat,nxn=df.shape[0],min_value=5,max_value=100)

        logger.info("Searching for optimal route...")
        logger.info("number of leads in file: {}".format(df.shape[0]))

        start,solution = explore(distance_mat,solver)
        end = time.time()

        logger.info("Order of {}:{}".format(mode,solution[0]))
        logger.info("Objective: {}".format(solution[1]))
        logger.info("{} solver timer: {:.2f} minutes".format(solver.upper(),(end-start)/60))

        fname = 'solution/{}_solution.csv'.format("defult")
        csv_export(data=df,order=solution[0],fout=fname)
        
        if plot is True:
            pname = 'defult'
            mapsplot_export(data=df,fout=pname,groupBy=pname)
    except:
        logger.error("Defult solver failed")
        raise


def google_distance_builder(data,fout,mode):
    """ google_distance_builder function
    building distance matrix using google distance matrix API, save matrix as fout.csv
    builder function should be call only ones for *.csv file (Google API $$$...)
    *** Google API key requiered ***
    params: 
    data = input data frame.
    fout = output .csv distance matrix file.
    """
    try:
        logger.critical("Using google key for distance matrix")
        # Google Maps API web service.
        Gmaps = google_client_register()

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


def geo_distance_builder(data,fout):
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


def pd_fill_diagonal(df_matrix, value=0): 
    mat = df_matrix.values
    n = mat.shape[0]
    mat[range(n), range(n)] = value
    return pd.DataFrame(mat)

def random_distance_builder(fout, nxn, min_value=1, max_value=100):
    """ random_distance_builder
    build random distance matrix size n X n with random leads name.
    params:
    fout      = output .csv distance matrix file.
    nxn       = size of matrix.
    min_value = minimum distance/duration value between leads.
    max_value = maximum distance/duration value between leads.
    """
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


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def csv_export(data,order,fout):
    """ csv_export function
    gets leads data from fin.csv, puts in fout.csv in order.
    params:
    fin   = input .csv file name, unordered. 
    order = new order permutation.
    fout  = output .csv file name, ordered by order arg. 
    """
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


# mplt.heatmap(df['latitude'], df['longitude'], df['value'],toFile=os.path.join(path,'1'))
# mplt.plot_markers(df,toFile=os.path.join(path,'2.jpeg'))
# mplt.density_plot(df['latitude'], df['longitude'],toFile=os.path.join(path,'3.jpeg'))
# mplt.polygons(df['latitude'], df['longitude'], df['postal_code'],toFile=os.path.join(path,'4.jpeg'))
# mplt.scatter(df['latitude'], df['longitude'], df['postal_code'],toFile=os.path.join(path,'5.jpeg'))
# mplt.polyline(df['latitude'], df['longitude'], closed=True,toFile=os.path.join(path,'6.jpeg'))
def mapsplot_export(data,fout,groupBy):
    """ mapsplot_export function

    params:
    data    = dataframe of points.
    fout    = name of export file.
    groupBy = 
    """
    try:
        path      = pathlib.Path(__file__).parent.parent  
        if groupBy == 'defult':
            # plot heat map on map
            file_name = 'solution/{}_density.png'.format(fout)
            mplt.density_plot(data['latitude'], data['longitude'],toFile=os.path.join(path,file_name))
        else:
            # plot covered area on map
            file_name = 'solution/{}_cover.png'.format(fout)
            mplt.polygons(data['latitude'], data['longitude'],data[groupBy],toFile=os.path.join(path,file_name))
        
        logger.info("mapsplot saved to: {}".format(file_name))
    except:
        logger.error("mapsplot export failed")



def google_client_register():
    """ google_client_register function
    Google Client API key to enable Google Distance Matrix calls.
    return:
    Gmaps = 
    """
    try:
        with open('data/api_key.txt', mode='r') as f:
            API_key = f.readline().strip()
            # Connect to googlemaps.
            Gmaps = googlemaps.Client(key=API_key)
            logger.info("Google distance matrix API_KEY successfully registered")
    except:
        logger.error("Google client register API_KEY failed")
        raise
    
    return Gmaps



def mapsplot_register():
    """ mapsplot_register function
    mapsplotlib Google Static Maps API key to enable queries to Google.
    """
    try:
        with open('data/api_key.txt', mode='r') as f:
            API_key = f.readline().strip()
            mplt.register_api_key(API_key)
            logger.info("Google static map API_KEY successfully registered")
    except:
        logger.error("mplt.register API_KEY failed")
        raise