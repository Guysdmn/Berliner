import numpy as np
import targetFunction as tfunc
import sys
from collections import OrderedDict

class GA_solver(object):

    # Initlize GA algorithm params
    def __init__(self, leads, population_size=100, evals=10**5,fin='csv/distance_matrix.csv'):

        self.pm          = 0.2
        self.pc          = 0.47
        self.pop_size    = population_size
        self.max_evals   = evals
        self.leads       = leads
        self.generations = 0
        self.imp_cntr    = 0
        self.fin         = fin
    
    # Order 2 crossover functoin, by given two permutation(parents), the function randomize interval from
    # one parent and build a new permutation with the randomize interval permutation and the other parent 
    # elements in their order.
    def crossover(self, perm1, perm2):
        N = len(perm1)
        h = np.random.randint(1, N)
        l = np.random.randint(0, h)
        
        p1 = perm1[l:h]
        p2 = []
        for i in perm2:
            if i not in p1:
                p2.append(i)
        newb = N - len(perm1[l:])
        
        if newb :
            child1 = np.concatenate((p2[:newb], p1[:], p2[newb:]), axis=0)
        else :
            child1 = np.concatenate((p1[:], p2[newb:]), axis=0)
        p2 = []
        p1 = perm2[l:h]
        for i in perm1:
            if i not in p1:
                p2.append(i)
        newb = N - len(perm2[l:])
        if newb :
            child2 = np.concatenate((p2[:newb], p1[:], p2[newb:]), axis=0)
        else :
            child2 = np.concatenate((p1[:], p2[newb:]), axis=0)
    
        return child1, child2


    # Radomize two diffrent indexes and swap their value in perm ( given permutation )
    def mutation(self, perm):
        N = len(perm)
        rnd = np.random.uniform(0,1)
        if rnd < self.pm:
            # ----------swap mutation:-----------
            h = np.random.randint(1, N - 1)
            l = np.random.randint(0, h)       
            result = np.copy(perm)
            result[l], result[h] = result[h], result[l]
            return result

        return perm

    # Select randomize number from 0 to the first quarter of the population number
    # this function must be called after population_list is sorted by the fitness function
    # by that, the chosen individual is one of the strongest ammong it's enviroment.
    def select(self, population_list):
        index = np.random.randint(0, int(self.pop_size*0.2))
        return population_list[index]

    # GA Algorithm implementation for the N Queens problem
    def run(self, seed=None):
        tf   = tfunc.TargetFunction(self.fin)
        fmax = sys.maxsize
        xmax    = []
        history = []
        population_list = []
        fitness_list    = []
        local_state     = np.random.RandomState(seed)
        for i in range(self.pop_size):
            population_list.append(np.random.permutation(self.leads))
            fitness_list.append(tf.fitness(perm=population_list[i]))
        fcurr_best = fmax = np.min(fitness_list)
        eval_cntr  = self.pop_size

        history.append(fmax)
        xmax = population_list[fitness_list.index(min(fitness_list))]

        while (eval_cntr < self.max_evals):

            newPopulationList = []
            # Crossover or mutation activation on half of the population
            for i in range(1, int(self.pop_size / 2)):
                parent1 = self.select(population_list)
                parent2 = self.select(population_list)
                if local_state.uniform(0, 1) < self.pc:
                    xChild1, xChild2 = self.crossover(parent1, parent2)
                else:
                    xChild1 = np.copy(parent1)
                    xChild2 = np.copy(parent2)

                xChild1 = self.mutation(xChild1)
                xChild2 = self.mutation(xChild2)
                newPopulationList.append(xChild1)
                newPopulationList.append(xChild2)

            # Elitist is a merge list of all the parents and their children.
            elitist = np.concatenate((newPopulationList, population_list), axis=0)
            # The whole big population is sorted before killing the 'weak' half of the population.
            elitist = sorted(elitist, key=tf.fitness)
            # Calculate number of different parents for next generation.
            elitset = np.copy(elitist)
            dic = OrderedDict()
            for x in elitset:
                key = (x[0],x[-1])
                if key not in dic:
                    dic[key] = x[1:-1]
                else:
                    val = dic[key]
                    dic[key] = [a+b for a,b in zip(val,x[1:-1])]
            elitset = [[k[0]] + v + [k[1]] for k,v in dic.items()]
            # Cut the population back to the original param-pop_size, killing the 'weak' population.
            population_list = elitist[:self.pop_size]
            # Generations counter.
            self.generations += 1
            # Best permutation in the current interation. 
            fcurr_best = tf.fitness(perm=population_list[0])
            # Count how many calls to fitness function has been made. 
            eval_cntr += 2*self.pop_size + 1
            # Check wheter the best new permutation is better then current best. 
            if fcurr_best < fmax:
                fmax = fcurr_best
                xmax = population_list[0]
                self.imp_cntr = 0
            else:
                self.imp_cntr += 1
            # Append best new permutation to history.
            history.append(fcurr_best)
            # Pring algorithm process.
            #print("Generation: " + str(self.generations) + " || Fitness: " + str(fmax) + " || Parents variety : " + str(len(elitset)))
            # If the optimum has found befor the algorithm iteration limit.
            if fmax == 0:
                break

            if self.imp_cntr == 200:
                break
        return xmax