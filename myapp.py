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

#squere test function
def sphere(x):
    return sum(np.power(np.e,-2*np.log(2)*np.power((x-0.08)/0.854,2))*np.power(np.sin(5*np.pi*((np.sign(x)*(np.abs(x))**.75)-.05)),6))    
    # return sum(np.power(np.sin(5*np.pi*x),6))
    # return sum(x**2)  #cost function

# Problem Definition
problem= structure()
##definition of cost function
problem.costfunc = sphere
##defenition of search space
problem.nvar = 5
problem.varmin = -10 #[-10, -10, -1, -5, 4]
problem.varmax = 10 #[10, 10, 1, 5, 10]

#GA Parameters
params=structure()
## maximum iteration
params.maxit = 100  
## the size of initial population
params.npop = 50
params.beta = 1
##population coefficient (if it is 1 then number of children equals to nember of parents) 
params.pc = 1
params.gamma = 0.1
## mutation parameters
params.mu = 0.1
params.sigma = 0.1

x=0
max = []
min = []
avg = []

while x < 10:
    #Run GA
    out = ga.run(problem, params)
    max.append(np.max(out.bestcost))
    min.append(np.min(out.bestcost))
    avg.append(sum(out.bestcost)/len(out.bestcost))
    x += 1

print("Max: " + str(np.max(max)))
print("Min: "+ str(np.min(avg)))
print("Average: "+ str(sum(avg)/10))
#Results
#plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0,params.maxit)
plt.xlabel("Iteration")
plt.ylabel("Best Cost")
plt.title("Genetic Algorithm (GA)")
plt.grid(True)
plt.show()
    