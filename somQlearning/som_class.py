#!/usr/bin/env python
"""
Class for Online/batch SOM for clustering/classification
Christos Mavridis & John Baras,
Electrical and Computer Engineering Dept.,
University of Maryland
"""

# Assumption: Map features in [0,1].

#%% Import Modules

import numpy as np
import math
from scipy.stats import multivariate_normal
    
#%% The Class
    
class gSOM:
    def __init__(self,y=[],ylabels=[],sigma=[],grid=[],
                 em_convergence=0.0001,effective_neighborhood=1e-3,
                 perturb_param=0.1,Bregman_phi='phi_Eucl',
                 convergence_loops=0,bb_init=0.1,bb_step=0.3):
        
        # EM Convergence parameters
        self.em_convergence = em_convergence 
        self.perturb_param = perturb_param
        self.effective_neighborhood = effective_neighborhood
        self.practical_zero = 1e-5
        
        # Bregman Divergence parameters
        self.Bregman_phi = Bregman_phi # 'phi_Eucl', 'phi_KL', 'phi_IS'
        self.phi = eval('self.'+Bregman_phi)
        
        # Training parameters
        self.convergence_loops = convergence_loops
        # self.run_batch = run_batch # if True, run originial DA algorithm
        # self.batch_size = batch_size # Stochastic/mini-batch Version  
        self.bb_init= bb_init # initial stepsize of stochastic approximation: 1/(bb+1)
        self.bb_step = bb_step # 0.5 # bb+=bb_step
    
        # Codevectors
        self.y = y
        self.ylabels = ylabels
        self.old_y = y
        self.sigma = sigma
        self.grid = grid
                
        # State Parameters
        self.K = len(self.y)
        # self.perturbed = False
        self.converged = False
        self.bb = self.bb_init
        self.em_steps = 0

    # Bregman Divergence Functions 
    
    # Phi functions: return phi, d_phi, dd_phi    
    def phi_Eucl(self,x):
        lenx = len(x) if hasattr(x, "__len__") else 1
        return (np.dot(x,x), 
                2*x, 
                np.diag(2*np.ones(lenx)) 
                )
    
    def phi_KL(self,x):
        lenx = len(x) if hasattr(x, "__len__") else 1
        # test = np.dot(x,np.where(x >self.practical_zero, np.log(x), 0))
        # if math.isnan(test):
        #     print(f'Inside KL: nan')
        return (np.dot(x,np.where(x > self.practical_zero, np.log(x), 0)), 
                np.ones(lenx) + np.where(x > self.practical_zero, np.log(x), 0),
                np.diag(np.ones(lenx)*np.where(x > self.practical_zero, 1/x, 0))
                )
    
    def phi_IS(self,x):
        lenx = len(x) if hasattr(x, "__len__") else 1
        return (-np.dot(np.log(x), np.ones(lenx)),
                -np.ones(lenx)/x,
                np.diag(np.ones(lenx)/x**2)
                )
    # Define Bregman Divergence:
    # return D, dy_D
    def BregmanD(self,x, y):
        phi = self.phi
        return (phi(x)[0] - phi(y)[0] - np.dot(phi(y)[1], x-y),
                - np.dot(phi(y)[2], x-y)
                )
    
    # Distortion Computation
        
    def distortion_hard(self, data, labels):
        # ignore labels
        y = self.y
        d_hard = 0.0
        for di in data:
            dists = [self.BregmanD(di,yj)[0] for yj in y]
            j = np.argmin(dists)
            d_hard += self.BregmanD(di,y[j])[0] 
        return d_hard     

    def classification_error(self, data, labels):
        y = self.y
        clabels = self.ylabels
        d_hard = 0.0
        for i in range(len(data)):
            di = data[i]
            dists = [self.BregmanD(di,yj)[0] for yj in y]
            # print([BregmanD(di,yj,phi)[0] for yj in y])
            j = np.argmin(dists)
            if clabels[j] != labels[i]:
                d_hard += 1 #BregmanD(di,centroids[j],phi)[0] 
        return d_hard       

    # to define: accuracy, typeI, typeII errors etc.
    
    # DA Algorithm Functions
    
    def overwrite_codevectors(self,new_y,new_ylabels): # new_y must be a list
        self.y = new_y
        self.ylabels = new_ylabels
        self.K = len(self.y) 
    
    def perturb(self):
        # insert perturbations of all effective yi
        for i in range(self.K):
            self.y[i] = self.y[i] + self.perturb_param*(np.random.rand(len(self.y[i]))-0.5)
           
    def lattice_weights(self, winner):
        winner_lattice = np.unravel_index(winner,self.grid)
        S = np.diag(self.sigma)
        gaussian = multivariate_normal(winner_lattice,S)
        wn = np.zeros(self.K)
        for i in range(self.K):
            wn[i] = gaussian.pdf(np.unravel_index(i,self.grid)) * \
                (2*np.pi)**2 * np.sqrt(np.linalg.det(S))
            
        # plt.imshow(wn)
        # plt.imshow(wn.reshape((-1,1)))
        return wn.reshape((-1,1))        
    
    def sa_step(self, datum, datum_label):
        self.old_y = self.y.copy()
        
        d = [self.BregmanD(datum,self.y[k])[0] 
                     for k in range(len(self.y))]
        
        winner = np.argmin(d) # w for winner
        lw = self.lattice_weights(winner)
        
        # Asynchronous SA
        for i in range(self.K):
            sign = 1 if datum_label == self.ylabels[i] else -1
            delta_d = self.BregmanD(datum,self.y[i])[1]
            self.y[i] = self.y[i] - 1/(self.bb+1)*sign*lw[i]*delta_d
        
        self.bb += self.bb_step 
        self.em_steps += 1
    
           
    def check_convergence(self):
        if self.convergence_loops>0: 
            if self.em_steps>=self.convergence_loops:
                self.converged=True
        else:
            if np.all([self.BregmanD(self.old_y[i],self.y[i])[0]<self.em_convergence 
                                                        for i in range(self.K)]):   
                self.converged=True
    
    # def find_effective_clusters(self):
    #     i=0
    #     while i<self.K:
    #         for j in reversed(np.arange(i+1,self.K)):
    #             if self.BregmanD(self.y[i],self.y[j])[0] < \
    #                 self.effective_neighborhood and self.ylabels[i]==self.ylabels[j]:
    #                 self.py[i] = self.py[i]+self.py[j]
    #                 self.sxpy[i] = self.y[i]*self.py[i]
    #                 self.y.pop(j)
    #                 self.ylabels.pop(j)
    #                 self.py.pop(j)
    #                 self.sxpy.pop(j)
    #                 self.K-=1
    #         i+=1
                    
    def train_som(self, datum, datum_label=0):
        
        # Termination Criteria
        if self.converged:
            pass
            # print('Converged.')
        else:
            
            # SA step
            self.sa_step(datum, datum_label)
            
            # Check Convergence
            self.check_convergence()
            
            # if self.converged:
                # Find effective clusters
                # self.find_effective_clusters() 
                
    
        