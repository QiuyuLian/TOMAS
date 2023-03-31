#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:04:27 2021

@author: qy
"""

import numpy as np
import scipy
from scipy.special import gammaln
#import time
#import pickle
import traceback
import pandas as pd
#import os, sys
import pyDIMM
from scipy import stats
import multiprocessing as mp
import anndata

#%% DMN optimization 

def dmn(adata, groupby, groups='all', tol=1e-3, maxiter=1000, subset=None, verbose=2, verbose_interval=10):
    '''
    Fit Dirichlet-Multinomial distribution with UMI counts of homo-droplet populations.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    groups : list of strings, optional
        Droplet categories to which DMN model shoudl be fitted with. It should be eithor 'all' or a list of cell type annotations specified in adata.obs[groupby].
        The default is 'all'.
    tol : float, optional
        The convergence threshold. The iteration will stop when the likelihood gain is below this threshold. The default is 1e-3.
    maxiter : int, optional
        The maximal number of iteration. The default is 1000.
    subset : int, optional
        Number of downsampled droplets to fit DMN. It is only recommended when you have excessive droplet numbers for homotypic droplet populations. The default is None.
    verbose : int, optional
        Enable verbose output. If 1 then it prints the current initialization and each iteration step. If greater than 1 then it prints also the log probability and time needed for each step.
        The default is 2.
    verbose_interval : int, optional
        Number of iteration done before the next print. The default is 10.

    Returns
    -------
    None.
    The optimized alpha vectors for each kind of droplet population is stroed in adata.varm['para_diri'].

    '''
    
    if groups=='all':
        groups = [d for d in adata.obs[groupby].unique() if len(d.split('_'))==1]
        
    counts_list = []
    vidx_list = []
    for group in groups:
    
        counts = adata[adata.obs[groupby]==group,:].X
        if isinstance(counts,scipy.sparse.csr.csr_matrix) or isinstance(counts, anndata._core.views.ArrayView):
            counts = counts.toarray()
        
        if not np.issubdtype(counts.dtype, np.integer):
            counts = counts.astype(int)
        
        vidx = counts.sum(0)!=0
        vidx_list.append(vidx)
        counts = counts[:,vidx]
        
        if subset is not None:
            if counts.shape[0] > subset:
                idx = np.random.choice(counts.shape[0], subset, replace=False)
                counts = counts[idx,:]
                
        counts_list.append(counts)
        
    n_cores = min(len(groups), int(mp.cpu_count()*0.8))
    
    pool = mp.Pool(n_cores)  
    result_compact = [pool.apply_async( _fitdmn, (counts_list[i], groups[i], tol,maxiter, verbose, verbose_interval) ) for i in range(len(groups))]
    pool.close()
    pool.join()
    
    alpha_out = [term.get() for term in result_compact] 
    
    #alpha_df = pd.DataFrame(np.array(alpha_out).T, index=adata.var_names, columns=groups)
    alpha_df = pd.DataFrame(np.ones([adata.n_vars,len(groups)])*1e-12, index=adata.var_names, columns=groups)
    for gidx,gval in enumerate(groups):
        alpha_df.loc[vidx_list[gidx],gval] = alpha_out[gidx]
    
    if 'para_diri' in adata.varm:
        for g in groups:
            adata.varm['para_diri'][g] = alpha_df[g]
    else:
        adata.varm['para_diri'] = alpha_df
    
    


def _fitdmn(counts,group,tol,maxiter,verbose,verbose_interval):
    
    dmm = pyDIMM.DirichletMultinomialMixture(n_components=1, 
                                             max_iter=maxiter, 
                                             n_init=1, 
                                             tol=tol,
                                             verbose=verbose,
                                             verbose_interval=verbose_interval).fit(counts)
    print(group+' is done!')
    return np.ravel(dmm.alphas)






#%% logNormal distribution optimization

def rm_outliers(x):
    iqr = stats.iqr(x)
    outlier_lb = np.quantile(x,0.25)-1.5*iqr
    outlier_ub = np.quantile(x,0.75)+1.5*iqr
    x_shrinkage = x[x > outlier_lb]
    x_shrinkage = x_shrinkage[x_shrinkage<outlier_ub]
    return x_shrinkage#,(outlier_lb,outlier_ub)


def logN_para(adata,logUMIby,groupby,groups='all',inplace=True):
    '''
    Fit logNormal distributions with UMI amounts of homo-droplet populations.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    logUMIby : str
        The key of total UMIs in log10 stored in adata.obs.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    groups : list of strings, optional
        Droplet categories to fit the total UMIs wtih logNormal distribution. It should be eithor 'all' or a list of cell type annotations specified in adata.obs[groupby].
        The default is 'all'.
    inplace : bool, optional
        If or not to store fitted parameters into adata. The default is True.

    Returns
    -------
    para : list
        Retuen mean and std of fitted logNormal distributions if inplace is False.
    '''
    if groups=='all':
        groups = adata.obs[groupby].unique()
    
    para = []
    for g in groups:
        m,s = stats.norm.fit(rm_outliers(adata.obs[logUMIby][adata.obs[groupby]==g]))
        para.append([m,s])

    if inplace:
        adata.uns['logUMI_para'] = pd.DataFrame(np.array(para),
                                                       index=groups,
                                                       columns=['mean','std'])
    else:
        return para







'''

class HiddenPrints:
    # Hide prints from pyDIMM in C
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _job_fitdmn(counts, group,c_version,maxiter,output):

    print('Start fitting '+group+' droplets. This may take a long time. Please wait...\n')
    if c_version:
        t0 = time.time()
        with HiddenPrints():
            dimm = pyDIMM.DIMM(observe_data=counts, n_components=1, print_log=True)
            dimm.EM(max_iter=maxiter, max_alpha_tol=1e-3, max_loglik_tol=1e-3, save_log=True)
        print('Finish fitting '+group+' droplets. Time cost:', time.time()-t0, 'seconds.')

        log = pd.read_csv('pyDIMM.log')
        log.to_csv(os.path.join(output,group+'.dmnlog.txt'))
        #os.remove('pyDIMM.log')
        alpha_vec = np.ravel(dimm.get_model()['alpha'])

    else:

        log = open(os.path.join(output,group+'.dmnlog.txt'),'w',buffering=1)    
        log.write('niter'+','+'alpha_sum'+','+'loglik'+','+'alpha_l2norm'+','+'alpha_l2norm_delta'+'\n')

        t0 = time.time()
        alpha_init = initialize_alpha(counts)
        alpha_vec = optimize_alpha(counts, alpha_init, log=log,maxiter=maxiter)
        print('Finish fitting '+group+' droplets. Time cost:', time.time()-t0, 'seconds.\n')
        log.close()
    
    return alpha_vec

'''

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
    
    minAlpha = 1e-100
    
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
    
    #record = {'alpha': [], 'LL': [], 'alpha_norm': [], 'delta_alpha':[]}
    
    logLik = sum(calculate_LL(alpha, Y))
    alpha_norm = np.linalg.norm(alpha)
    
    #record['alpha'].append(alpha)
    #record['LL'].append(logLik)
    #record['alpha_norm'].append(alpha_norm)
    
    alpha_hits = []
    
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
            
            #record['alpha'].append(alpha)
            #record['LL'].append(logLik)
            #record['alpha_norm'].append(alpha_norm)
            #record['delta_alpha'].append(delta_alpha)
            iteration += 1
            #if log is None:
            #    print('iter '+str(iteration)+': precision = '+str(sum(alpha)) + ', LL = ' + str(logLik))
            #else:
            log.write(str(iteration)+','+str(sum(alpha)) +','+str(logLik) +','+str(alpha_norm)+','+str(delta_alpha)+ '\n')
            
            # if delta_alpha is less than delta_alpha_Tol in continuous keeptimes, then break
            alpha_hits.append( delta_alpha < delta_alpha_Tol)
            if not np.all(alpha_hits):
                alpha_hits = []
        #elif len(alpha_hits) >= keeptimes:
        '''
        if log is None:
            print('terminated after '+str(iteration)+' iterations.')
        else:
            log.write('terminated after '+str(iteration)+' iterations.\n')
            #break        
        '''
    except Exception as e:
        #if log is None:
        print('repr(e):\n',repr(e))
        print('traceback.print_exc():')
        traceback.print_exc()
        '''
        else:
            log.write('repr(e):\t'+repr(e)+'\n')
            log.write('traceback.print_exc():\n'+str(traceback.print_exc())+'\n')
        '''
    return alpha
    



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


