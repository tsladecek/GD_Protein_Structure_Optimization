#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Optimization
"""
# %%
import torch
import numpy as np
from distributions import *
from structure import Structure
from copy import copy
import pickle
import argparse
import os
import matplotlib.pyplot as plt

# %%
C_vdW = 2 * 1.7

def steric_repulsion(dmap):
    """Steric repulsion term between C-beta atoms"""
    
    sr = 0
    d = ((C_vdW ** 2 - dmap ** 2) ** 2) / C_vdW
    for i in range(len(dmap) - 1):
        for j in range(i + 1, len(dmap)):
            if dmap[i, j] < C_vdW:
                sr += d[i, j]
    return sr


def NLLLoss(structure, 
            normal=True, 
            steric=False):
    """
    Loss Function consisting of two potentials:
        distance potential
        torsion angle potential
    
    distance potential is the log of a probability of a distance value
    from a distribution to which a cubic spline is fitted
    
    angle potential 
    """
    
    x = torch.linspace(2, 22, 31)
    xtorsion = torch.linspace(-np.pi, np.pi, 36)
    loss = 0
        
    # DISTANCE POTENTIAL
    distance_map = structure.G()
    if normal or normal == 'True' or normal == 'T':
        for i in range(len(distance_map) - 1):
            for j in range(i + 1, len(distance_map)):
                mu, sigma, s = structure.normal_params[0, i, j],\
                               structure.normal_params[1, i, j],\
                               structure.normal_params[2, i, j]
                
                if mu <= structure.distance_threshold:
                    loss += torch.log(max(torch.tensor(0.0001), 
                                      normal_distr(distance_map[i, j], mu, sigma, s)))
     
    else:  # fit cubic spline to histograms
        for i in range(len(distance_map) - 1):
            for j in range(i + 1, len(distance_map)):
                loss += torch.log(max(torch.tensor(0.001),
                                      interp(x,
                                             structure.distogram[:, i, j],
                                             min(torch.tensor(22), distance_map[i, j]))))
                
    # Steric clashes term
    if steric or steric == 'True' or steric == 'T':
        loss -= steric_repulsion(distance_map) 
    
    return -loss


def gd_psr(structure,
           normal=True,
           iterations=100,
           iteration_start=0,
           lr=1e-3,
           gradient_scaling='sddiv',  # one of ['sddiv', 'normal', 'absmaxdiv']
           momentum=0,
           nesterov=False,
           steric=False,
           verbose=-1):
    
    """
    Gradient Descent Protein Structure Realization
    
    Input:
        structure          : object of class Structure
        normal             : bool, whether (scaled) normal distribution should be fitted to the distograms\n
        iterations         : int, iterations of the gradient descent algorithm\n
        iteration_start    : int, iteration start integer
        lr                 : float, learning rate\n
        gradient_scaling   : str, type of gradient scaling. Either 'sddiv' (division by standard deviation), 'normal' for standard normalization or "absmaxdiv" for division by the absolute maximum value
        momentum           : float, momentum parameter\n
        nesterov           : bool, default=False, Nesterov Momentum\n
        steric             : bool, steric clashes potential term for loss function
        verbose            : how often should the program print info about losses. Default=iterations/20\n
        
    Output:
        best_structure: structure with minimal loss\n
        min_loss      : loss of the best structure\n
        history       : history of learning
    """
    
    initial_lr = copy(lr)
    if verbose == -1:
        verbose = int(np.ceil(iterations / 20))
    
    # OPTIMIZE, OPTIMIZE, OPTIMIZE
    history = []
    min_loss = np.inf
    
    if momentum > 1 or momentum < 0:
        print('Momentum parameter has to be between 0 and 1')
        return
    
    # initialize V for momentum
    V = torch.zeros((len(structure.torsion)))
    
    for i in range(iteration_start, iteration_start + iterations):
        
        if structure.torsion.grad is not None:
            structure.torsion.grad.zero_()
        
        if nesterov is True or nesterov == 'True' or nesterov == 'T':
            structure.torsion = (structure.torsion + momentum * V).detach().requires_grad_()
            
        L = NLLLoss(structure, normal, steric=steric)
        
        loss_minus_th_loss = L.item() - structure.min_theoretical_loss
        
        if loss_minus_th_loss < min_loss:
            best_structure = copy(structure)
            min_loss = loss_minus_th_loss
        
        L.backward()
        
        #print(structure.torsion.grad[:5])
        if gradient_scaling == 'normal':
            # normalize gradients
            structure.torsion.grad = (structure.torsion.grad - torch.mean(structure.torsion.grad)) / torch.std(structure.torsion.grad)
        elif gradient_scaling == 'sddiv':
            structure.torsion.grad = structure.torsion.grad / torch.std(structure.torsion.grad)
        elif gradient_scaling == 'absmaxdiv':
            # gradients inside range from -1 to 1
            structure.torsion.grad = structure.torsion.grad / torch.max(torch.abs(structure.torsion.grad))
        
        # Implementing momentum
        V = momentum * V - lr * structure.torsion.grad
        
        structure.torsion = (structure.torsion + V).detach().requires_grad_()
        
        if verbose != 0:
            if i % verbose == 0 or i == iterations - 1:
                print('Iteration {:03d}, Loss: {:.3f}'.format(i, loss_minus_th_loss))
            
        if L.item() == np.inf or L.item() == -np.inf or L.item() is None:
            print('Loss = inf')
            return
        
        history.append([i, loss_minus_th_loss])
    
    return best_structure, min_loss, history


def optimize(domain,
             distogram_path,
             seq_path,
             random_state=1,
             normal=True,
             output_dir=None,
             iterations=200, 
             distance_threshold=18,
             restarts=5,
             lr=1e-3, 
             lr_decay=0.1,
             gradient_scaling='sddiv',
             momentum=0,
             nesterov=False,
             steric=False,
             verbose=-1):
    
    """
    Gradient Descent (restarted) Protein Structure Realization 
    
    Input:
        domain             : str, domain name\n
        distogram_path     : path to distogram\n
        seq_path           : path to fasta sequence\n
        random_state       : int\n
        normal             : bool, whether (scaled) normal distribution should be fitted to the distograms\n
        output_dir         : output directory\n
        iterations         : int, iterations of the gradient descent algorithm\n
        distance_threshold : only distances (distogram) less than "distance_threshold" are used for optimization
        restarts           : int, number of times the learning rate is altered after "iterations" pass
        lr                 : float, learning rate\n
        gradient_scaling   : str, type of gradient scaling. Either 'sddiv' (division by standard deviation), 'normal' for standard normalization or "absmaxdiv" for division by the absolute maximum value
        momentum           : float, momentum parameter\n
        nesterov           : bool, default=False, Nesterov Momentum\n
        steric             : bool, steric clashes potential term for loss function
        verbose            : how often should the program print info about losses. Default=iterations/20\n
        
    Output:
        dictionary:
            beststructure : structure with minimal loss\n
            loss          : loss of the best structure\n
            history       : history of learning
            
        if output_dir is not None, the dictionary is pickled
    """
    
    history = []
    initial_lr = lr
    
    structure = Structure(domain=domain, 
                          distogram_path=distogram_path, 
                          seq_path=seq_path, 
                          random_state=random_state, 
                          normal=normal,
                          distance_threshold=distance_threshold)
    
    for r in range(restarts):
        s, l, h = gd_psr(structure=structure, 
                         normal=normal,
                         iterations=iterations, 
                         iteration_start=r*iterations,
                         lr=lr,
                         gradient_scaling=gradient_scaling,
                         momentum=momentum,
                         nesterov=nesterov,
                         steric=steric,
                         verbose=verbose)
        
        lr = lr_decay * lr
        structure = s
        history.extend(h)
    
    d = {'beststructure':structure, 'loss':l, 'history':np.array(history)}
    if output_dir is not None:
        
        with open('{}/{}_{:d}_{:.3f}_{:.1f}_pred.pkl'.format(output_dir, domain, random_state, initial_lr, momentum), 'wb') as f:
            pickle.dump(d, f)
    else:
        return d

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Structure optimization with restarts')
    
    parser.add_argument('-d', '--domain', required=True, help='Domain Name')
    parser.add_argument('-dp', '--distogram_path', required=True, help='Path to Distogram')
    parser.add_argument('-sp', '--seq_path', required=True, help='Path to fasta sequence')
    
    parser.add_argument('-rs', '--randomstate', type=int, metavar='', required=False, help='Random Seed. Default = 1', default=1)
    parser.add_argument('-n', '--normal', metavar='', required=False, help='Fit normal distr to distograms. Default = True. If False, 3rd degree spline is fitted', default='True')
    parser.add_argument('-o', '--outputdir', metavar='', required=False, help='Output Directory', default='./')
    parser.add_argument('-i', '--iterations', type=int, metavar='', required=False, help='Number of iterations. Default = 200', default=200)
    parser.add_argument('-dt', '--distancethreshold', type=float, metavar='', required=False, help='only distances (real) less than "distance_threshold" are used for optimization. Default = 18A', default=18)
    parser.add_argument('-r', '--restarts', type=int, metavar='', required=False, help='Number of restarts of the Optimization process from the best previous state with decreased learning rate. Default = 5', default=5)
    parser.add_argument('-lr', '--learningrate', type=float, metavar='', required=False, help='Learning rate. Default = 0.01', default=0.01)
    parser.add_argument('-ld', '--lrdecay', type=float, metavar='', required=False, help='Learning rate decay parameter after each restart. Default = 0.1', default=0.1)
    parser.add_argument('-gr', '--gradientscaling', metavar='', required=False, help='What type of gradient scaling should be applied to the gradient. Options: sddiv (division by standard deviation), normal (standard normalization), absmaxdiv (division by the absolute maximum value). Default = sddiv', default='sddiv')
    parser.add_argument('-m', '--momentum', type=float, metavar='', required=False, help='Momentum parameter. Default = 0', default=0.0)
    parser.add_argument('-nm', '--nesterov', metavar='', required=False, help='Nesterov Momentum. Default = False', default='False')
    parser.add_argument('-sc', '--stericclashes', metavar='', required=False, help='Steric clashes potential term for loss function. Default = False', default='False')
    parser.add_argument('-v', '--verbose', type=int, metavar='', required=False, help='How often should the program print info about losses. Default = iterations / 20', default=-1)

    args = parser.parse_args()


    optimize(domain=args.domain,
             distogram_path=args.distogram_path,
             seq_path=args.seq_path,
             random_state=args.randomstate,
             normal=args.normal,
             output_dir=args.outputdir,
             iterations=args.iterations, 
             distance_threshold=args.distancethreshold,
             restarts=args.restarts,
             lr=args.learningrate, 
             lr_decay=args.lrdecay,
             gradient_scaling=args.gradientscaling,
             momentum=args.momentum,
             nesterov=args.nesterov,
             steric=args.stericclashes,
             verbose=args.verbose)
            
