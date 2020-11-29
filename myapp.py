# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:08:30 2020

@author: sshaf
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import csv

#parameters
pM = 0.1
pC = 0.1
trial_count = 10
benchmark = 1 #use 1 or 4

#squere test function
def test(x):
    if benchmark == 4:
        return sum(np.power(np.e,-2*np.log(2)*np.power((x-0.08)/0.854,2))*np.power(np.sin(5*np.pi*((np.sign(x)*(np.abs(x))**.75)-.05)),6))    
    elif benchmark == 1:
        return sum(np.power(np.sin(5*np.pi*x),6))
    else:
        return False

# Problem Definition
problem= structure()
##definition of cost function
problem.costfunc = test
##defenition of search space
problem.nvar = 2
problem.varmin = 0 #[-10, -10, -1, -5, 4]
problem.varmax = 1 #[10, 10, 1, 5, 10]

#GA Parameters
params=structure()
## maximum iteration
params.maxit = 100  
## the size of initial population
params.npop = 50
params.beta = 1
##population coefficient (if it is 1 then number of children equals to nember of parents) 
params.pc = 1
params.gamma = pC
## mutation parameters
params.mu = pM
params.sigma = 0.1

x=0
max = []
min = []
avg = []

while x < trial_count:
    #Run GA
    out = ga.run(problem, params,"sharing")
    max.append(np.max(out.bestcost))
    min.append(np.min(out.worstcost))
    avg.append(sum(out.bestcost)/len(out.bestcost))
    print("Run " + str(x+1) + " done")
    x += 1

print("Max: " + str(np.max(max)))
print("Min: "+ str(np.min(avg)))
print("Average: "+ str(sum(avg)/10))
#Results
#plt.plot(out.bestcost)
plt.semilogy(out.bestcost, label = "best")
plt.plot(out.worstcost, label = "worst")
plt.xlim(0,params.maxit)
plt.xlabel("Iteration")
plt.ylabel("Best Cost")
plt.title("Genetic Algorithm (GA)")
plt.grid(True)
plt.legend()
plt.show()
    