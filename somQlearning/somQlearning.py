#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:55:31 2020

@author: cm
"""

#%%

import pickle
import random
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from matplotlib import cm 

# import sys
# sys.path.append('../')

import cartpole
from som_class import gSOM

plt.ioff() # turn off interactive mode: only show figures with plt.show()
plt.close('all')

# random seed
rs=1
np.random.seed(rs)
random.seed(rs) 

#%% Hyperparameters 

# Hyperparameters
gamma_Q = 0.9
epsilon = 0.2 #epsilon_0
epercent = 0.8 # epsilon = epercent * epsilon
# aa=10 # a = aa/(aa+t)
# bb=1 # bb/(t*ln(t)+1000*bb)
aa_init=0.1 
aa_step = 0.3 

nnc=5
nc = [nnc, nnc, nnc, nnc] # [x, x_dot, theta, theta_dot]
box = [1.0, 4.0, 1.0, 4.0]
# box = [0.5, 2.0, 0.5, 2.0]

low_D=False
# low_D=True
# box = [2.5, 10, 5, 10]
# nc = [5, 5, 2, 2]

train_loops = 7
test_loops = 3
vq_epochs = 2
vq_mod = 2

time_steps = 500 # maximum number of timesteps per try


# State Aggregation
bb_init=1000.8 # initial stepsize of stochastic approximation: 1/(bb+1)
bb_step = 10.5 # 0.5 # bb+=bb_step

em_convergence = 0.0001 # need to choose automatically if not in [0,1]
perturb_param = 0.1 
effective_neighborhood = 0.1
Bregman_phi = 'phi_Eucl' # 'phi_Eucl', 'phi_KL', 'phi_IS'
convergence_loops = 1e10 # if >0 forces loops to be done

# SOM
SOM = True
ss=3
sigma = [ss, ss, ss, ss]
spercent = 0.6

# save results
save_results = True
results_file = './results/test.pkl'

# plot results
plot_state_space = True
plot_training_curve = True
plot_V3D = False
plot_V = False
animation = False

#%% Environment

env = cartpole.CartPoleEnv()
env_seed=env.seed(rs)
env.action_space.np_random.seed(rs)

#%% Initial Clusters

cluster_list=[]

if not low_D:
    for i in range(len(nc)):
        cluster_list.append(np.linspace(-box[i],box[i],nc[i])) 
    
    cluster_size = np.prod(nc)
    clusters = np.zeros((cluster_size,len(nc)))
    n=0
    for i in range(nc[0]):
        for j in range(nc[1]):
            for k in range(nc[2]):
                for l in range(nc[3]):
                    clusters[n][0] = cluster_list[0][i]
                    clusters[n][1] = cluster_list[1][j]
                    clusters[n][2] = cluster_list[2][k]
                    clusters[n][3] = cluster_list[3][l]
                    n+=1
else:
    for i in range(2):
        cluster_list.append(np.linspace(-box[i+2],box[i+2],nc[i+2])) 

    cluster_size = np.prod(nc[2:])
    clusters = np.zeros((cluster_size,len(nc[2:])))
    n=0
    for i in range(nc[2]):
        for j in range(nc[3]):
            clusters[n][0] = cluster_list[0][i]
            clusters[n][1] = cluster_list[1][j]
            n+=1

clusters=[clusters[i] for i in range(clusters.shape[0])]
cluster_labels = [0 for i in range(cluster_size)]

# SOM

if not low_D:
    grid=nc
else:
    grid=nc[2:]
    sigma = sigma[2:]
som = gSOM(y=clusters,ylabels=cluster_labels,sigma=sigma,grid=grid,
     em_convergence=em_convergence,effective_neighborhood=effective_neighborhood,
     perturb_param=perturb_param,Bregman_phi=Bregman_phi,
     convergence_loops=convergence_loops,bb_init=bb_init,bb_step=bb_step)

#%% Plot State Space 
                
if plot_state_space:
    # for plotting
    clusters2= np.linspace(-0.25,0.25,nc[2]) 
    clusters3= np.linspace(-2,2,nc[3]) 
    
    clustersx = np.zeros((nc[2]*nc[3],2))
    n=0
    for i in range(len(clusters2)):
        for j in range(len(clusters3)):
            clustersx[n][0] = clusters2[i]
            clustersx[n][1] = clusters3[j]
            n+=1
            
    fig = plt.figure(facecolor='white') 
    # plt.ylim(-5, 5)
    # plt.ylim(-15, 15)
    plt.plot([clustersx[i][0] for i in range(clustersx.shape[0])],
              [clustersx[i][1] for i in range(clustersx.shape[0])],'ko')
    plt.plot([clusters[i][0] for i in range(len(clusters))],
              [clusters[i][1] for i in range(len(clusters))],'b*')
    if not low_D:
        plt.plot([clusters[i][2] for i in range(len(clusters))],
                  [clusters[i][3] for i in range(len(clusters))],'r*')
    else:
        plt.plot([clusters[i][0] for i in range(len(clusters))],
                  [clusters[i][1] for i in range(len(clusters))],'r*')        
    
    plt.savefig('./state_space/0.png')
    plt.close(fig)

#%% Q-Learning 

Q = list(np.zeros(cluster_size*env.action_space.n).reshape((-1,env.action_space.n)))

def alpha(t):
    return 1/(1+aa_init+t*aa_step)
    # return 1.0/aa 

# def safe_ln(x):
#     if x <= 0:
#         return 0
#     return np.log(x)

# def beta(t):
#     return bb/(t*safe_ln(t)+1e3*bb)
#     # return 1.0/bb                    

# find state representative in the set of codevectors
def q_state(state):
    s = state
    eucl_dist = np.sum((s-clusters)**2,axis=1)
    qs = np.argmin( eucl_dist )
    return np.int_(qs)

def eGreedy(state, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore action space
    else:
        return np.argmax(Q[q_state(state)]) # Exploit learned values

def update_Q(t, state, action, new_state, cost):
    Qold = Q[q_state(state)][action]
    Qnew = np.max(Q[q_state(new_state)]) 
    # Qsarsa = Q[tuple(q_state(new_state))][eGreedy(new_state,epsilon)]
    Q[q_state(state)][action] = Qold + \
                alpha(t) * ( cost + gamma_Q *Qnew - Qold )
   
#%% Training Loop

training_avg=[]

# for all training loops
for k in range(train_loops):
    
    avg = 0
    
    # for 100 epochs
    for i in range(100):
        
        state = env.reset()
        if low_D:
            state = state[2:]
        # restart stepsizes
        if SOM:
            som.bb = bb_init
            som.converged=False
            
        # until failure and up to time_steps
        for t in range(time_steps):
            
            # pick next action
            action = eGreedy(state,epsilon)
            
            # observe new state
            new_state, cost, terminate, info = env.step(action)
            if low_D:
                new_state = new_state[2:]
                
            # Update clustering
            if (t+1)%((k+1)*vq_mod)==0: # (i+1)%((k+1)*vq_mod)==0:
                
                if SOM:
                    som.train_som(state)
                    clusters=som.y
            
            # Update Q
            update_Q(t, state, action, new_state, cost)
            state=new_state
            
            if terminate:
                # print("Episode finished after {} timesteps".format(t+1))
                break
    
        # compute average over 100 repeats    
        # avg = avg * i/(i+1) + (t+1)/(i+1) #which is equivalent with
        avg = avg + 1/(i+1) * (t+1-avg)               
    
    training_avg.append(avg)
    print(f'{k+1}-th hundred (e={round(epsilon,2)}) : Average timesteps: {avg}')
    
    fig = plt.figure(facecolor='white') 
    # plt.ylim(-5, 5)
    # plt.ylim(-15, 15)
    plt.plot([clustersx[i][0] for i in range(clustersx.shape[0])],
              [clustersx[i][1] for i in range(clustersx.shape[0])],'ko')
    plt.plot([clusters[i][0] for i in range(len(clusters))],
              [clusters[i][1] for i in range(len(clusters))],'b*')
    if not low_D:
        plt.plot([clusters[i][2] for i in range(len(clusters))],
              [clusters[i][3] for i in range(len(clusters))],'r*')
    else:
        plt.plot([clusters[i][0] for i in range(len(clusters))],
                 [clusters[i][1] for i in range(len(clusters))],'r*')  
    plt.savefig(f'./state_space/{k+1}.png')
    plt.close(fig)

    if k+1>=vq_epochs:
        SOM = False
        
    if SOM:
        sigma = [sigma[i]*spercent for i in range(len(sigma))]
        som.sigma = sigma
            
    epsilon = epercent*epsilon

#%% Testing Loop

if True:
    epsilon_test=1e-6
    testing_avg=[]
    avg = 0
    
    for k in range(test_loops):
        for i in range(100):
            state = env.reset()
            if low_D:
                state = state[2:]
            for t in range(3*time_steps):
                action = eGreedy(state,epsilon_test)
                new_state, cost, terminate, info = env.step(action)
                if low_D:
                    new_state = new_state[2:]
                # max_state = np.maximum(np.abs(state),np.abs(new_state))
                if k<1:
                    update_Q(t, state, action, new_state, cost)
                state=new_state
                if terminate:
                    break
            # avg = avg * i/(i+1) + (t+1)/(i+1)
            avg = avg + 1/(i+1) * (t+1-avg)
        testing_avg.append(avg)
        print(f'Average Number of timesteps: {avg}')

#%% Save results to file 

if save_results:    
    my_results = [training_avg, testing_avg]
                
    if results_file != '':
        with open(results_file, mode='wb') as file:
            pickle.dump(my_results, file) 

#%% Training Curve
    
if plot_training_curve:
    
    fig = plt.figure(facecolor='white')
    plt.title('Training Curve')
    plt.plot(np.arange(len(training_avg))+1,training_avg, label='Training Averages')
    plt.plot(len(training_avg)+np.zeros(len(testing_avg))+1,testing_avg,'r*',
                      label='Testing Averages')
    plt.xlabel('Hundreds of episodes')
    plt.ylabel('Average number of timesteps')
    plt.legend()

    plt.show()
    
#%% V values 3D plot

if plot_V3D:
    
    V = np.max(Q,axis=1) 
    
    # create a new figure for plotting 
    fig = plt.figure(facecolor='white') 
    # and set projection as 3d 
    ax = fig.gca(projection='3d')
    
    # defining x, y, z co-ordinates 
    mesh_points = 101
    box2 = 0.25
    box3 = 2
    s2 = np.linspace(-box2, box2, mesh_points)
    s3 = np.linspace(-box3, box3, mesh_points)
    S2, S3 = np.meshgrid(s2, s3)
    
    # Plot the surface.
    V_plot = np.zeros((mesh_points,mesh_points))
    for i in range(mesh_points):
        for j in range(mesh_points):
            qs = q_state([0,0,s2[i],s3[j]]) 
            V_plot[i,j] = V[qs]
            
    surf = ax.plot_surface(S2, S3, V_plot, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    # Customize x,y axis
    # ax.set_xlim(-box2, box2)
    ax.set_xlabel(r'$\theta$')
    # ax.set_ylim(-box3, box3)
    ax.set_ylabel(r'$\dot\theta$')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()

#%% V values imshow

if plot_V and low_D:

    V = np.max(Q,axis=2) 
    
    # create a new figure for plotting 
    fig = plt.figure(facecolor='white') 
    # and set projection as 3d 
    plt.imshow(V_plot,cmap=cm.coolwarm)
    plt.colorbar(shrink=0.5, aspect=5)
    
    plt.show()

#%% Animation

if animation:

    epsilon=1e-6
    avg = 0
    
    for i in range(1):
        state = env.reset()
        for t in range(500):
            env.render()
            # print(state)
            action = eGreedy(state,epsilon)
            new_state, cost, terminate, info = env.step(action)
            update_Q(t, state, action, new_state, cost)
            state=new_state
            if terminate:
                print("Episode finished after {} timesteps".format(t+1))
                break
        avg = avg * i/(i+1) + (t+1)/(i+1)
    print(f'Average Number of timesteps: {avg}')
    env.close()