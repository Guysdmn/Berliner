import pandas as pd
from itertools import tee

class TargetFunction():
    def __init__(self,fin):
        self.df = pd.DataFrame(pd.read_csv(fin))
        self.df = self.df.drop(self.df.columns[[0]], axis=1)
        self.fd = self.df.values
    
    def pairwise(self,iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def fitness(self,perm): 
        duration = 0
        for i1,i2 in self.pairwise(perm):
            duration += self.fd[i1][i2]
        duration += self.fd[perm[-1]][perm[0]]
        return duration