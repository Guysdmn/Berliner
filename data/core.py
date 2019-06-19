import sys
import pandas as pd
import numpy as np
import googlemaps
from geopy import distance
import time
from itertools import islice, tee

""" distance_builder function
reads relevant columns ['name','latitude','longitude'] from fin.csv file 
and building distance matrix using google distance matrix API, save matrix as fout.csv
builder function should be call only ones for leads.csv file (Google API $$$...)
params: 
fin = input .csv leads file.
fout = output .csv distance matrix file.
"""
def distance_builder(fin,fout):
    # Data frame init with required columns.
    df = pd.read_csv(fin,usecols = ['name','latitude','longitude'])

    # Google Maps API web service.
    try:
        with open('data/api_key.txt', mode='r') as f:
            API_key = f.readline().strip()
    except:
        raise IOError

    try:
        # Connect to googlemaps.
        Gmaps = googlemaps.Client(key=API_key)

        # Init distance matrix - will be used to store calculated distances and times.
        disMat   = pd.DataFrame(0,columns=df.name.unique(), index=df.name.unique())
        apiCalls = 0

        # Start building timer.
        start = time.time()

        # Loop through each row in the data frame.
        for (i1, row1) in df.iterrows():
            # Assign latitude and longitude as origin points.
            LatOrigin  = row1['latitude']
            LongOrigin = row1['longitude']
            origin     = (LatOrigin,LongOrigin)

            # Loop through unvisited paths in the data frame (decrease API calls $$$).
            for (i2, row2) in islice(df.iterrows(),i1):
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
                    disMat[row1['name']][row2['name']] = 10^3
                    disMat[row2['name']][row1['name']] = 10^3
                    continue
                
                # Pass origin and destination variables to distance_matrix googlemaps function.
                result = Gmaps.distance_matrix(origin, destination, mode='walking')
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
        print("-----------------------------------------------------------------------")
        print("Built distane matrix in: {:.2f}".format((end-start)/60) , "minutes with {}".format(apiCalls), "Google API calls")
        print("Saved to: ",fout)
        print("-----------------------------------------------------------------------")
    except:
        raise IOError

""" csv_export function
gets leads data from fin.csv, puts in fout.csv in order.
params:
fin   = input .csv file name, unordered. 
order = new order permutation.
fout  = output .csv file name, ordered by perm order. 
"""
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def csv_export(fin,order,fout):
    # Read original .csv file.
    try:
        df = pd.read_csv(fin)
    except:
        raise IOError
    # Create new data frame.
    orderd = pd.DataFrame(columns=['name', 'owners', 'types','opening_hours_tuesday','rating','disqualification_reason',
                                   'phone_number','email','is_qualifed','latitude','longitude','place_id'])
    # Append to new data frame in order.
    for idx in order:
        orderd = orderd.append(df.iloc[idx])

    # Save ordered to *.csv file.
    orderd.to_csv(fout, index=False)


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

def random_distance_builder(fout, nxn=100, min_value=1, max_value=100):
    try:
        # Create leads name.
        names = ['s' + str(x) for x in range(1,nxn+1)]
        # Create random value distance matrix.
        df = pd.DataFrame(np.random.randint(min_value,max_value,size=(nxn, nxn)),columns=names,index=names)
        # Fill diagonal with 0 value.
        pd_fill_diagonal(df,0)
        # Check "geo distance" like.
        df = df.apply(lambda x: [y if y <= 80 else (max_value*2) for y in x])
        # Save to fout.csv file.
        df.to_csv(fout)
    except:
        raise