#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:04:27 2021

@author: qy
"""

import numpy as np
import scipy
from scipy.special import gammaln
import time
import pickle
import traceback


def estimateAlpha(counts, filename, saveOpt=True):
    '''
    Fit a Dirichlet-Multinomial distribution given UMI counts of a homo-droplet population.

    Parameters
    ----------
    counts : numpy.ndarray
        UMI matrix with rows as genes and columns as droplets.
        dense or sparse matrix both works.
    filename : str
        path and name to create an optimization log.
        its form should be '/path_to_save_the_file/name_of_the_file' without suffix.
    saveOpt : bool, optional
        save the intermediate variables during the optimization process or not. The default is False.

    Returns
    -------
    alpha_vec : numpy.ndarray
        the optimized Dirichlet parameter. 
        alpha_vec.shape = (num_genes,)

    '''

    log = open(filename+'.log.txt','w',buffering=1)
    #log.write(filename+'\n')
    
    log.write('data loading ...'+'\n')
    if isinstance(counts,scipy.sparse.csr.csr_matrix):
        counts = counts.toarray()
    
    # counts = np.loadtxt('./sim_data/counts_'+file+'_'+w_set+'.csv', delimiter=',')
    
    log.write('optimization starts.'+'\n')
    t0 = time.time()
    alpha_init = initialize_alpha(counts)
    alpha_vec, record = optimize_alpha(counts, alpha_init, log=log,maxiter=3000)
    log.write('optimization ends, taking '+str(time.time()-t0) +' seconds.'+'\n')
    
    #log.write('visualizing ...'+'\n')
    #visualize_alpha_optimization(record, filename=filename)
        
    log.write('record saved.')
    log.close()
    
    if saveOpt:
        f = open( filename+'.dmn.pickle','wb')
        pickle.dump(record,f)
        f.close()

    return alpha_vec, record

        

def initialize_alpha(Y):

    gene_tot = np.ravel(np.sum(Y, axis=0))
    lib_size = np.ravel(np.sum(Y, axis=1))
    
    G = sum(gene_tot > 0)
    #C = sum(lib_size > 0)
    
    new_Y = Y[lib_size > 0, :]
    new_Y = new_Y[:, gene_tot > 0] 
    
    p_mat = ( new_Y.T/lib_size[lib_size > 0] ).T
    p_mean = p_mat.mean(0)
    p_var = p_mat.var(0)
    p_var[p_var == 0] = np.nanmean(p_var)
    
    alpha_sum = np.exp( sum(np.log(p_mean[:-1]*(1-p_mean[:-1])/p_var[:-1] - 1))/(G-1) )
    
    alpha = np.array([0.0]*Y.shape[1])
    alpha[gene_tot > 0] = alpha_sum * p_mean
    alpha[alpha == 0.0] = 1e-6

    return alpha
    


def optimize_alpha(Y, alpha, **para):
    
    minAlpha = 1e-200
    
    maxiter = para.get('maxiter',2000)
    likTol = para.get('likTol',1e-2)
    log = para.get('log',None)
    delta_alpha_Tol = para.get('alpTol',1e-1)
    keeptimes =  para.get('ntimes',100)
    
    delta_LL = 100.0
    delta_alpha = 20
    iteration = 0
    
    lib_size = np.ravel(np.sum(Y, axis=1))
    C,G = Y.shape
    
    record = {'alpha': [], 'LL': [], 'alpha_norm': [], 'delta_alpha':[]}
    
    logLik = sum(calculate_LL(alpha, Y))
    alpha_norm = np.linalg.norm(alpha)
    
    record['alpha'].append(alpha)
    record['LL'].append(logLik)
    record['alpha_norm'].append(alpha_norm)
    
    alpha_hits = []
    
    if log is None:
        print('iter '+str(iteration)+': precision = '+str(sum(alpha)) + ', LL = ' + str(logLik))
    else:
        log.write('iter '+str(iteration)+': precision = '+str(sum(alpha)) + ', LL = ' + str(logLik) + '\n')
    
    
    try:
    
        while( ((delta_LL > likTol) or len(alpha_hits) <  keeptimes) and (iteration < maxiter)):
            
            # calculate new alpha
            alphaNew = np.zeros(alpha.shape)
            den = sum(lib_size/(lib_size -1 + alpha.sum() ))
            for gidx in range(G):
                dataG = (( Y[:,gidx] + minAlpha)/(Y[:,gidx] + minAlpha - 1 + alpha[gidx]))
                num = dataG.sum()
                alphaNew[gidx] = alpha[gidx] * num / den
            
            # calculate new logLik
            LL_tmp = calculate_LL(alphaNew, Y)
            newLogLik  = LL_tmp.sum()
            
            # calculate diff to check convergence
            delta_LL = np.abs( (newLogLik - logLik)/logLik*100.0 )
            delta_alpha = np.linalg.norm(alphaNew - alpha)
            # update 
            alpha = alphaNew
            alpha[alpha <= 0] = minAlpha
            logLik = newLogLik
            
            alpha_norm = np.linalg.norm(alpha)
            
            record['alpha'].append(alpha)
            record['LL'].append(logLik)
            record['alpha_norm'].append(alpha_norm)
            record['delta_alpha'].append(delta_alpha)
            iteration += 1
            if log is None:
                print('iter '+str(iteration)+': precision = '+str(sum(alpha)) + ', LL = ' + str(logLik))
            else:
                log.write('iter '+str(iteration)+': precision = '+str(sum(alpha)) + ', LL = ' + str(logLik) + '\n')
            
            # if delta_alpha is less than delta_alpha_Tol in continuous keeptimes, then break
            alpha_hits.append( delta_alpha < delta_alpha_Tol)
            if not np.all(alpha_hits):
                alpha_hits = []
        #elif len(alpha_hits) >= keeptimes:
            
        if log is None:
            print('terminated after '+str(iteration)+' iterations.')
        else:
            log.write('terminated after '+str(iteration)+' iterations.\n')
            #break        
            
    except Exception as e:
        if log is None:
            print('repr(e):\n',repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
        else:
            log.write('repr(e):\t'+repr(e)+'\n')
            log.write('traceback.print_exc():\n'+str(traceback.print_exc())+'\n')

    return alpha, record
    



def lgammaVec(inVec):
    outVec = gammaln(inVec)
    idx_inf = np.isinf(outVec)
    outVec[idx_inf] = 709.1962086421661
    # lnGamma(1e-308) = 709.1962086421661, lnGamma(1e-309) = inf
    # if val is closer than 1e-308 to 0, e.g. 1e-309, inf will be returned 
    # we would truncate it to a certain number: here it's lnGamma(1e-308)
    return outVec




def calculate_LL(alpha, Y):
    
    lib_size = np.ravel(np.sum(Y, axis=1))
    C = Y.shape[0]
    lgammaAlpha = lgammaVec(alpha)
    alphaSum = alpha.sum()    
    
    LL_tmp = np.array([0]*C)
    num1 = np.array([0]*C)#np.zeros((C,1))
    num2 = np.array([0]*C)#np.zeros((C,1))
    for i in range(C):
        dataAlphaTmp = Y[i,:] + alpha
        num1[i] = sum(lgammaVec(dataAlphaTmp) - lgammaAlpha)
        num2[i] = gammaln(alpha.sum()) - gammaln( alphaSum + lib_size[i] )
        LL_tmp[i] = num1[i] + num2[i] 
    
    return LL_tmp


