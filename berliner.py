import sys
sys.path += ['data','solver','csv']
import numpy as np
import pandas as pd
import time

from solver import orsolver, acosolver #, orsolvertw, gasolver
import core

def solve(fin,fout):
    try:
        input_file = open(fin,'r')
        input_file.close()
    except:
        print("ERROR: {} file not found".format(fin))
        sys.exit()

    distance_mat = 'csv/distance_matrix.csv'

    # build distance matrix
    try:
        #core.distance_builder(fin,distance_mat)
        core.random_distance_builder(fout=distance_mat,nxn=10,min_value=5,max_value=100)
    except IOError:
        print("ERROR: data/api_key.txt not found")
        sys.exit()
    except:
        print("ERROR: distance matrix error")
        sys.exit()
    
    try:
        print('-----------------------START----------------------------')
        disf = pd.DataFrame(pd.read_csv(distance_mat))
        disf = disf.drop(disf.columns[[0]], axis=1)
        fd   = disf.values.copy()
        
        # initialize solvers
        #GA_solver   = gasolver.GA_solver(fin=fout,leads=len(fd))
        OR_solver   = orsolver.OR_solver(distance_matrix=distance_mat)
        #OR_solvertw = orsolvertw.OR_solver_TW(fin=fout)
        ACO_solver  = acosolver.ACO_solver(Graph=fd,seed=345)

        # solve
        # OR_tools
        start = time.time()
        or_perm = OR_solver.run()
        end = time.time()
        print("OR solver timer: {:.2f}".format((end-start)/60), "minutes")
        # ACO
        start = time.time()
        aco_perm = ACO_solver.run()
        end = time.time()
        print("\nACO solver solution:\nObjective: {:.2f}".format(aco_perm[1]), "minutes")
        print("ACO solver timer: {:.2f}".format((end-start)/60), "minutes")

        print('-----------------------FINNISH---------------------------')
    except:
        print("ERROR: solver error")
        sys.exit()
    try:
        core.csv_export(fin=fin,order=or_perm,fout=fout)
    except IOError:
        print("ERROR: {} file not found".format(fin))
        sys.exit() 
    except:
        print("ERROR: error exporting solution to {}".format(fout))
        sys.exit()
    

if __name__ == "__main__":
    # berliner.solve(fin.csv, fout.csv)
    
    fin  = sys.argv[1] #'csv/*.csv'
    fout = sys.argv[2] #'csv/distance_matrix*.csv'
    solve(fin,fout)
    

