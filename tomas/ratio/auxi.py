#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:04:27 2021

@author: qy
"""



import numpy as np

import pandas as pd
import os
from scipy.special import loggamma, gammaln
from scipy.stats import multinomial, dirichlet, beta, norm
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
import copy
import traceback


#%% estiamte alpha 

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
    





def visualize_alpha_optimization(record,fig_title,output = None):
    
    filename = fig_title
    
    fig, axs =plt.subplots(2,2,figsize=(9,6),dpi=128)
    
    plt.subplot(2,2,1)
    term = record['LL']
    plt.scatter(range(len(term)-2), term[2:], s=1)
    plt.xlabel('iteration')
    plt.ylabel('logLikelihood')
    plt.title('logLikelihood',fontsize=14)
    #plt.show()
    
    plt.subplot(2,2,2)
    term = record['alpha_norm']
    plt.scatter(range(len(term)), term, s=1)
    #plt.show()
    plt.xlabel('iteration')
    plt.ylabel('alpha l2-norm')
    plt.title('alpha l2-norm',fontsize=14)
    
    plt.subplot(2,2,3)
    term = record['delta_alpha']
    plt.scatter(range(len(term)), term, s=1)
    #plt.show()
    plt.xlabel('iteration')
    plt.ylabel('delta alpha')
    plt.title('delta alpha',fontsize=14)
    
    plt.subplot(2,2,4)
    term = [sum(x) for x in record['alpha']]
    plt.scatter(range(len(term)), term, s=1)
    #plt.show()
    plt.xlabel('iteration')
    plt.ylabel('sum alpha')
    plt.title('sum alpha',fontsize=14)
    
    
    fig.tight_layout()
    tit = fig.suptitle(filename,fontsize=16)
    fig.subplots_adjust(top=0.85)
    if output is not None:
        plt.savefig(os.path.join(output, filename+'.jpg'),bbox_extra_artists=(tit,), bbox_inches='tight')
    #plt.show()
    
    
    
#%% estimate w

def get_LL_W(P1, P2, W, X, P1_ll, P2_ll, k = None):
    
    if k is not None:
        print(k,'th process ...')
    
    LL_w_x = np.zeros((len(W), len(X)))

    for w_idx in range(len(W)):
        w = W[w_idx]
        LL_w_x[w_idx] = get_LL_w(P1, P2, w, X, P1_ll, P2_ll)
        
    return LL_w_x


def get_LL_w(P1, P2, w, X, P1_ll, P2_ll):
    num_p = len(P1)
    num_x = len(X)
    
    LL_sum = np.array([0.0]*num_x)
    N = X.sum(1)
    
    P_wsum = P1*w + P2*(1-w)
    
    for i in range(num_p):
        p_tmp = P_wsum[i]
        LL_sum += get_LL_w_p(p_tmp, w, X, N) + P1_ll[i] + P2_ll[i]

    return LL_sum


def get_LL_w_p(p_tmp, w, X, N):
    
    p_tmp[p_tmp == 0] = 1e-10
    p_tmp_sum = sum(p_tmp)
    p_tmp = p_tmp/p_tmp_sum
    mult_ll = X*np.log(p_tmp)

    return mult_ll.sum(1) # multinomial.logpmf(X, N, p_tmp)    


def get_P_valid(P):
    
    P_valid = np.zeros(P.shape)
    for i in range(P.shape[0]):
        p_tmp = P[i]
        p_tmp[p_tmp == 0] = 1e-10
        p_tmp_sum = sum(p_tmp)
        p_tmp = p_tmp/p_tmp_sum
        
        P_valid[i,:] = p_tmp
        
    return P_valid



def sampling_to_estimate_W(X, Alpha1, Alpha2, num_p, w_step = 0.01):
    
    W = np.arange(0, 1+w_step, w_step)
    
    t = time.time()
    P1 = dirichlet.rvs(Alpha1, size = num_p)
    P2 = dirichlet.rvs(Alpha2, size = num_p)
    print('Time of sampling ' + str(num_p) + ' P: ', time.time() - t)
    
    
    t = time.time()
    P1_valid = get_P_valid(P1)
    P1_ll = [sum((Alpha1-1)*np.log(p)) for p in P1_valid]
    
    P2_valid = get_P_valid(P2)
    P2_ll = [sum((Alpha2-1)*np.log(p)) for p in P2_valid]
    print('Time of normalizing P and calculating LL of Diri: ', time.time() - t)
    
    
    K = int(num_p/5)
    
    t0 = time.time()
    pool = mp.Pool()  
    LL_compact =[pool.apply_async( get_LL_W, (P1_valid[5*k:5*(k+1)], P2_valid[5*k:5*(k+1)], W, X, P1_ll[5*k:5*(k+1)], P2_ll[5*k:5*(k+1)], k) ) for k in range(K)]
    pool.close()
    pool.join()
    print('multiprocessing: ', time.time()-t0)
    
    
    LL_flat = [term.get() for term in LL_compact]
    
    w_best_track = np.zeros((K, X.shape[0]))
    
    LL_sum_tmp = LL_flat[0]
    w_best_track[0] = W[LL_sum_tmp.argmax(0)]
    for i in range(1,K):
        LL_sum_tmp += LL_flat[i]
        w_best_track[i] = W[LL_sum_tmp.argmax(0)]
    
    
    plt.hist(w_best_track[0],30)
    plt.show()
    
    plt.hist(w_best_track[-1],30)
    plt.xlabel('w',fontsize=14)
    plt.ylabel('freq',fontsize=14)
    plt.title('Estimate w with '+str(num_p)+' sampled p', fontsize=16)
    plt.show()
    
    w_alpha, w_beta,_,_ = beta.fit(w_best_track[-1], floc=0.,fscale=1.)
    print('Estimated mean of w: ',w_alpha/(w_alpha+w_beta))

    return w_best_track
    

#%% simulation for debug

def generate_alpha(precision, num_genes):
    diri_p = np.random.beta(0.005, 1, num_genes) + 1e-8
    diri_p = diri_p/sum(diri_p)
    return precision * diri_p
    


def obtain_library_size(n_cell, l_m, l_std):
    log_ls = np.random.normal(l_m, l_std, n_cell)
    return (10**log_ls).astype(int)


obtain_counts = lambda n,p_vec: np.random.multinomial(n,p_vec)



def generate_counts(alpha_vec, n_cells=200, l_m=2.1, l_std=0.2):

    p_mat = np.random.dirichlet(alpha_vec, size = n_cells)

    library_sizes = obtain_library_size(n_cells, l_m, l_std)
    counts = np.array([obtain_counts(library_sizes[i], p_mat[i]) for i in range(n_cells)])

    return counts



def generate_doublets(alpha_vec_1, alpha_vec_2, w_mix=0.1, n_cells_mix=200, l_m_mix=2.1, l_std_mix=0.2):

    #w_mix = 0.1
    #n_cells_mix = 200
    # assume cell-type-x component in doublet follow the same Diri with cell-type-x singlets
    p_mix_1 = np.random.dirichlet(alpha_vec_1, size = n_cells_mix)
    p_mix_2 = np.random.dirichlet(alpha_vec_2, size = n_cells_mix)
    p_mat_mix = p_mix_1*w_mix + p_mix_2*(1-w_mix)
    
    #l_m_mix = 4.12191 + np.log10(5)
    #l_std_mix = 0.26832408
    
    library_sizes_mix = obtain_library_size(n_cells_mix, l_m_mix, l_std_mix)
    counts_mix = np.array([obtain_counts(library_sizes_mix[i], p_mat_mix[i]) for i in range(n_cells_mix)])

    #X = counts_mix
    return counts_mix, p_mix_1, p_mix_2




def estimate_w_given1p(x, p1, p2, w_step=0.01):
    
    W = np.arange(0, 1+w_step, w_step)
    tmp = []
    for w_idx in range(len(W)):
        
        w = W[w_idx]
        P_wsum = p1*w + p2*(1-w)
        P_wsum[P_wsum==0] = 1e-10
        sum_term = P_wsum.sum()
        P_wsum = P_wsum/sum_term
        tmp.append( sum(x*np.log(P_wsum)))
    #w_best = W[np.argmax(tmp)]
    
    return W[np.argmax(tmp)]




def estimate_w_with1p(x,Alpha1, Alpha2, mode='sampling', return_p = False, w_step=0.01):
    
    W = np.arange(0, 1+w_step, w_step)

    if mode == 'sampling':
        p1_sampling = np.random.dirichlet(Alpha1,1)[0,:] #Alpha1/Alpha1.sum() 
        p2_sampling = np.random.dirichlet(Alpha2,1)[0,:]  #Alpha2/Alpha2.sum()
    elif mode == 'mean_p':
        p1_sampling = Alpha1/Alpha1.sum() 
        p2_sampling = Alpha2/Alpha2.sum()
        
    tmp = []
    for w_idx in range(len(W)):
        
        w = W[w_idx]
        P_wsum = p1_sampling*w + p2_sampling*(1-w)
        P_wsum[P_wsum==0] = 1e-10
        sum_term = P_wsum.sum()
        P_wsum = P_wsum/sum_term
        tmp.append( sum(x*np.log(P_wsum)))
    
    w_best = W[np.argmax(tmp)]
    if return_p:
        return w_best, p1_sampling, p2_sampling
    else:
        return w_best




def estimate_w_with1p_resampling(x,Alpha1, Alpha2, return_p = False, w_step=0.01):
    
    W = np.arange(0, 1+w_step, w_step)
    w_best = 0.0
    
    times = 0
    while w_best*(1-w_best) == 0 and times<=1000:
        
        times += 1
        p1_sampling = np.random.dirichlet(Alpha1,1)[0,:] #Alpha1/Alpha1.sum() 
        p2_sampling = np.random.dirichlet(Alpha2,1)[0,:]  #Alpha2/Alpha2.sum()
            
        ll = []
        for w_idx in range(len(W)):
            
            w = W[w_idx]
            p_wsum = p1_sampling*w + p2_sampling*(1-w)
            p_wsum[p_wsum==0] = 1e-10
            sum_term = p_wsum.sum()
            p_wsum = p_wsum/sum_term
            ll.append( sum(x*np.log(p_wsum)))
        
        w_best = W[np.argmax(ll)]
    
    if return_p:
        return w_best, p1_sampling, p2_sampling
    else:
        return w_best




#%% estimate p 

def estimate_p_mle_x(x, alpha1, alpha2, logR_m, logR_std, num_sampling):
    
    logR_sampling = np.random.normal(logR_m, logR_std, num_sampling)
    logR_ll = norm.logpdf(logR_sampling, logR_m, logR_std)
    w_sampling = 1/(2**logR_sampling + 1)
    
    p1_sampling_mat = np.random.dirichlet(alpha1, num_sampling)
    p2_sampling_mat = np.random.dirichlet(alpha2, num_sampling)
    
    p1_sampling_mat = get_P_valid(p1_sampling_mat)
    p2_sampling_mat = get_P_valid(p2_sampling_mat)
    
    diri_ll_1 = np.matmul((alpha1 - 1).reshape(1,-1), np.transpose(np.log(p1_sampling_mat)))
    diri_ll_2 = np.matmul((alpha1 - 1).reshape(1,-1), np.transpose(np.log(p1_sampling_mat)))
    
    p_mix_mat = np.transpose( np.transpose(p1_sampling_mat) * w_sampling + np.transpose(p2_sampling_mat) * (1-w_sampling) )
    mult_ll = np.matmul(x.reshape(1,-1), np.transpose(np.log(p_mix_mat)))
    
    ll = diri_ll_1[0] + diri_ll_2[0] + mult_ll[0] + logR_ll
    
    idx = np.argmax(ll)
    
    return p1_sampling_mat[idx], p2_sampling_mat[idx]




def estimate_p_mle(X, alpha1, alpha2, logR_m, logR_std, num_sampling = 10):
    
    p1_list = []
    p2_list = []
    for x in X:
        p1_tmp,p2_tmp =  estimate_p_mle_x(x, alpha1, alpha2, logR_m, logR_std, num_sampling)
        p1_list.append(list(p1_tmp))
        p2_list.append(list(p2_tmp))

    return np.array(p1_list), np.array(p2_list)



#%% 

def calculateLL_x_real(alpha_vec_1, alpha_vec_2, w_mix, l_m_mix, l_std_mix):
    """
    Generate a doublet and return its logLikelihood

    """
    p_mix_1 = np.random.dirichlet(alpha_vec_1, size = 1)
    p_mix_2 = np.random.dirichlet(alpha_vec_2, size = 1)
    p_mix = p_mix_1*w_mix + p_mix_2*(1-w_mix)
    
    p_mix_1 = get_P_valid(p_mix_1)[0]
    p_mix_2 = get_P_valid(p_mix_2)[0]
    p_mix = get_P_valid(p_mix)[0]

    n = obtain_library_size(1, l_m_mix, l_std_mix)
    x = obtain_counts(n, p_mix) 
    
    diri_ll_1 = sum((alpha_vec_1 - 1)*np.log(p_mix_1)) 
    diri_ll_2 = sum((alpha_vec_2 - 1)*np.log(p_mix_2))
    mult_ll = sum(x*np.log(p_mix)) 
    
    logR_ll = norm.logpdf(np.log2(1/w_mix-1), np.log2(1/w_mix-1), 0.77)
    # ll_real = diri_ll_1 + diri_ll_2 + mult_ll
    ll_real = diri_ll_1 + diri_ll_2 + mult_ll + logR_ll
    
    
    #return x, ll_real, (diri_ll_1, diri_ll_2, mult_ll)
    return x, ll_real, (diri_ll_1, diri_ll_2, mult_ll, logR_ll), (p_mix_1, p_mix_2)




def sweep_w_maxLL(x, Alpha1, Alpha2, p1_sampling, p2_sampling, logR_m=np.log2(9), logR_std = 0.1):
    
    diri_ll_1 = sum((Alpha1 - 1)*np.log(p1_sampling)) 
    diri_ll_2 = sum((Alpha2 - 1)*np.log(p2_sampling))
    
    logR_sampling = np.random.normal(logR_m,logR_std,100)
    W = 1/(2**logR_sampling+1)
    logR_ll = norm.logpdf(logR_sampling, logR_m,logR_std)
    # W = beta.rvs(2,10,0,1,100)
    # logR_ll = beta.logpdf(W, 2,10,0,1)
    # W = np.arange(0, 1+w_step, w_step)
    mult_ll = []
    for w_idx in range(len(W)):
        w = W[w_idx]

        p_wsum = p1_sampling*w + p2_sampling*(1-w)
        p_wsum[p_wsum==0] = 1e-10
        sum_term = p_wsum.sum()
        p_wsum = p_wsum/sum_term
        mult_ll.append( sum(x*np.log(p_wsum)) )

    ll_max_idx = np.argmax(np.array(mult_ll)+logR_ll) # given p1-p2 pair, diri_ll_1 and diri_ll_1 are constant
    ll_max = mult_ll[ll_max_idx] + logR_ll[ll_max_idx] + diri_ll_1 + diri_ll_2
    
    return W[ll_max_idx], ll_max, mult_ll[ll_max_idx], diri_ll_1, diri_ll_2, logR_ll[ll_max_idx]




def estimate_w_mle(x, Alpha1, Alpha2, num_p = 1000, return_w_ll = False, **para):
    
    #fix_p = para.get('fix_p',False)
    return_p = para.get('return_p', False)
    p1_sampling_mat = para.get('p1_fix',None)
    p2_sampling_mat = para.get('p2_fix',None)
    logR_m = para.get('logR_m', None)
    logR_std = para.get('logR_std', None)
        
    if p1_sampling_mat is None:
        p1_sampling_mat = np.random.dirichlet(Alpha1,num_p)
        p2_sampling_mat = np.random.dirichlet(Alpha2,num_p)
        
        p1_sampling_mat = get_P_valid(p1_sampling_mat)
        p2_sampling_mat = get_P_valid(p2_sampling_mat)

    w_list = []
    ll_list = []
    d1_list = []
    d2_list = []
    mu_list = []
    logR_list = []
    for pidx in range(num_p):
        p1_sampling = p1_sampling_mat[pidx]
        p2_sampling = p2_sampling_mat[pidx]
        
        w_tmp, ll, mult_ll, diri_ll_1, diri_ll_2, logR_ll = sweep_w_maxLL(x, Alpha1, Alpha2, p1_sampling, p2_sampling, logR_m, logR_std)
        
        d1_list.append(diri_ll_1)
        d2_list.append(diri_ll_2)
        w_list.append(w_tmp)
        mu_list.append(mult_ll)
        ll_list.append(ll)
        logR_list.append(logR_ll)
        
    if return_w_ll:
        if return_p:
            return w_list[np.argmax(ll_list)], np.array([ll_list, w_list, d1_list, d2_list, mu_list, logR_list]), (p1_sampling_mat, p2_sampling_mat)
        else:
            return w_list[np.argmax(ll_list)], np.array([ll_list, w_list, d1_list, d2_list, mu_list, logR_list])
    
    else:
        return w_list[np.argmax(ll_list)]




def compare_samplingP_ll(ngenes, npvecs, alpha1, alpha2, w_set, l_m_mix, l_mix_std=0.2, return_ll = False, return_x_p = False, **para):
    """
    Visualize likelihood of multiple p1-p2 pairs sampled for deconvoluting a certin droplet.
    
    There are two test modes: 
    Mode 1: simulate one doublet and calculate its generating likelihood, run one-round estimateion and return sampled p1-p2 pairs.
    
        ll_dic, x, (p1_mat, p2_mat) = compare_samplingP_ll(ngenes, npvecs, alpha1, alpha2 , w_set, l_m_mix, return_ll=True, return_x_p=True, logR_m = logR_m, logR_std = 0.77)
        highlight_samplingP_ll(ll_dic)
        
    Mode 2: fix x and sampled p1-p2 pairs, update logR_m one time to estimate w again.
        
        ll_dic_01 = compare_samplingP_ll(ngenes, npvecs, alpha1, alpha2, w_set, l_m_mix, return_ll=True, x_fix=x, real_ll_set=ll_dic['ll_real_set'], p1_fix = p1_mat, p2_fix = p2_mat,logR_m = np.log2(1/ll_dic['w_x'] - 1), logR_std = 0.77)
        highlight_samplingP_ll(ll_dic_01)
    
    """
    x = para.get('x_fix',None)
    ll_real_set = para.get('real_ll_set',None)
    p1_mat = para.get('p1_fix', None)
    p2_mat = para.get('p2_fix', None)
    logR_m = para.get('logR_m', None)
    logR_std = para.get('logR_std', None)
    
    if x is None:
        x, ll_real_complete, ll_real_set, (p_real_1, p_real_2) = calculateLL_x_real(alpha1, alpha2, w_mix=w_set, l_m_mix=l_m_mix, l_std_mix=0.2)
    else:
        ll_real_complete = np.sum(ll_real_set)
        
    if return_x_p:
        w_x, ll_w_arr, (p1_mat, p2_mat) = estimate_w_mle(x, alpha1, alpha2, num_p = npvecs, return_w_ll = True, return_p=return_x_p, logR_m=logR_m, logR_std = logR_std)
    else:
        w_x, ll_w_arr = estimate_w_mle(x, alpha1, alpha2, num_p = npvecs, return_w_ll = True, p1_fix = p1_mat, p2_fix = p2_mat, logR_m=logR_m,  logR_std = logR_std)
    
    ll_max = [np.max(ll_w_arr[0,:nidx]) for nidx in range(1,npvecs+1)]
    
    #mult_delta = ll_w_arr[4] - ll_set[2]
    #print('max MultNormal LL, w = ',ll_w_arr[1][np.argmax(mult_delta)])
    
    if return_ll:
        
        ll_dic = {'alpha1':alpha1,
                  'alpha2':alpha2,
                  'diri1': ll_w_arr[2], 
                  'diri2': ll_w_arr[3], 
                  'mult': ll_w_arr[4], 
                  'logR': ll_w_arr[5], 
                  'completell': ll_w_arr[0],
                  'accumulatedMaxll': ll_max,
                  'w':ll_w_arr[1],
                  'll_real_complete':ll_real_complete,
                  'll_real_set':ll_real_set,
                  #'p_real_1': p_real_1,
                  #'p_real_2': p_real_2,
                  'w_x':w_x,
                  'w_set': w_set,
                  'ngenes': ngenes,
                  'npvecs': npvecs}
        #np.array([ll_w_arr[2] - ll_triplet[0], ll_w_arr[3] - ll_triplet[1], ll_w_arr[4] - ll_triplet[2], ll_w_arr[0] - ll_real])
        if return_x_p:

            return ll_dic, x, (p1_mat, p2_mat)
        else:
            return ll_dic
        
    
    


def highlight_samplingP_ll(ll_dic, hidx=None):
    """
    hight a point in results of sampling p1-p2 pairs for deconvoluting a droplet

    """
    p_mean_1 = ll_dic['alpha1'] / sum(ll_dic['alpha1'] )
    diri_meanpdf_1 = sum((ll_dic['alpha1'] - 1)*np.log(p_mean_1)) 

    p_mean_2 = ll_dic['alpha2'] / sum(ll_dic['alpha2'] )
    diri_meanpdf_2 = sum((ll_dic['alpha2'] - 1)*np.log(p_mean_2))
    
    ll_max = ll_dic['accumulatedMaxll']
    ll_real_set = ll_dic['ll_real_set']
    ll_real_complete = ll_dic['ll_real_complete']
    
    fig,axs = plt.subplots(3, 3, figsize=(12,10), dpi=128)
    
    plt.subplot(3,3,1)
    plt.plot(range(len(ll_max)), ll_max)
    plt.plot([0,len(ll_max)], [ll_real_complete,ll_real_complete],color='gray',linestyle='dashed')
    plt.title('max ll', fontsize=16)
    
    plt.subplot(3,3,2)
    plt.scatter(ll_dic['w'], ll_dic['diri1'] - ll_real_set[0],s=1)
    if hidx is not None:
        plt.plot(ll_dic['w'][hidx], ll_dic['diri1'][hidx] - ll_real_set[0], marker = '*', color='red')
    plt.plot([0,1],[0,0],color='gray',linestyle='dashed')
    plt.plot([0,1],[diri_meanpdf_1-ll_real_set[0], diri_meanpdf_1-ll_real_set[0]],color='red',linestyle='dashed')
    plt.title('Diri 1', fontsize=16)
    #plt.show()
    
    plt.subplot(3,3,3)
    plt.scatter(ll_dic['w'], ll_dic['diri2']  - ll_real_set[1],s=1)
    if hidx is not None:
        plt.plot(ll_dic['w'][hidx], ll_dic['diri2'][hidx] - ll_real_set[1], marker = '*', color='red')    
    plt.plot([0,1],[0,0],color='gray',linestyle='dashed')
    plt.plot([0,1],[diri_meanpdf_2-ll_real_set[1], diri_meanpdf_2-ll_real_set[1]],color='red',linestyle='dashed')
    plt.title('Diri 2', fontsize=16)
    #plt.show()

    plt.subplot(3,3,4)
    plt.scatter(ll_dic['w'], ll_dic['diri1'] + ll_dic['diri2'] - ll_real_set[0] - ll_real_set[1],s=1)
    plt.plot([0,1],[0,0],color='gray',linestyle='dashed')
    plt.plot([0,1],[diri_meanpdf_1+diri_meanpdf_2-ll_real_set[0]-ll_real_set[1], diri_meanpdf_1+diri_meanpdf_2-ll_real_set[0]-ll_real_set[1]],color='red',linestyle='dashed')
    plt.title('Diri 1 and 2', fontsize=16)
    
    plt.subplot(3,3,5)
    plt.scatter(ll_dic['w'], ll_dic['mult']  - ll_real_set[2],s=1)
    if hidx is not None:
        plt.plot(ll_dic['w'][hidx], ll_dic['mult'][hidx] - ll_real_set[2], marker = '*', color='red')
    plt.plot([0,1],[0,0],color='gray',linestyle='dashed')
    plt.title('Mult', fontsize=16)
    #plt.show()

    plt.subplot(3,3,8)
    plt.scatter(ll_dic['w'], ll_dic['logR'] - ll_real_set[3],s=1)
    if hidx is not None:
        plt.plot(ll_dic['w'][hidx], ll_dic['logR'][hidx] - ll_real_set[3], marker = '*', color='red')
    plt.plot([0,1],[0,0],color='gray',linestyle='dashed')
    plt.title('logR', fontsize=16)
    #plt.show()
    
    plt.subplot(3,3,6)
    plt.scatter(ll_dic['w'], ll_dic['completell'] - ll_real_complete, s=1)
    if hidx is not None:
        plt.plot(ll_dic['w'][hidx], ll_dic['completell'][hidx] - ll_real_complete, marker = '*', color='red')
    plt.plot([0,1],[0,0],color='gray',linestyle='dashed')
    plt.title('complete ll', fontsize=16)
    #plt.show()
    
    fig.suptitle('w_set = '+str(ll_dic['w_set'])+', w_out = ' +str(ll_dic['w_x'])+', '+str(ll_dic['ngenes'])+' genes, '+str(ll_dic['npvecs'])+' p1-p2 pairs',fontsize=16)
    # fig.suptitle('w = '+str(w)+', w_1p = '+str(w_x_1p) +', w_mle100p = '+str(w_x_mle_100p),fontsize=16)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    plt.show()
    




def W_mle_generatingP(alpha_vec_1, alpha_vec_2, w_mix=0.1, n_cells_mix=200, l_m_mix=2.1, l_std_mix=0.2, w_step=0.01):
    """
    Deconvolute w with doublet-generating p1-p2 pair

    Parameters
    ----------
    alpha_vec_1 : TYPE
        DESCRIPTION.
    alpha_vec_2 : TYPE
        DESCRIPTION.
    w_mix : TYPE, optional
        DESCRIPTION. The default is 0.1.
    n_cells_mix : TYPE, optional
        DESCRIPTION. The default is 200.
    l_m_mix : TYPE, optional
        DESCRIPTION. The default is 2.1.
    l_std_mix : TYPE, optional
        DESCRIPTION. The default is 0.2.
    w_step : TYPE, optional
        DESCRIPTION. The default is 0.01.

    Returns
    -------
    w_best : TYPE
        DESCRIPTION.

    """
    X, p_mix_1, p_mix_2 = generate_doublets(alpha_vec_1, alpha_vec_2, w_mix, n_cells_mix, l_m_mix, l_std_mix)

    w_best = [estimate_w_given1p(X[i],p_mix_1[i],p_mix_2[i],w_step) for i in range(len(X))]
    
    return w_best








def job_estimateW(Alpha_arr, w_set, ngenes, npvecs, ave_gcounts_arr, nsamples= 200, w_err=0.001, output=None, w_init = 0.5):
    """
    calculate w_MLE in the model: p1~Dir(a1), p2~Dir(a2), R~logNormal, R = 1/w-1

    Parameters
    ----------
    Alpha_arr : arr
        np.array([alpha1, alpha2]).
    w_set : float between 0 and 1
        true w.
    ngenes : int
        num of genes.
    npvecs : int
        num of sampled p1-p2 pairs to estimate w for each doublet.
    ave_gcounts_arr : arr
        genes counts summed up to estimate library size.
    nsamples : int, optional
        number of doublets. The default is 200.
    w_err : float, optional
        threshold for w estimation. The default is 0.001.
    output : str, optional
        path to save results. if none, results will not be saved. The default is None.

    Returns
    -------
    w_list : list
        estimated w for each doublet.

    """
    try:
    
        #w_set = round(w_settings[k],1)
        #ngenes = ngenes_settings[k]
        #print(1)
        alpha1 = Alpha_arr[0,:ngenes]
        alpha2 = Alpha_arr[1,:ngenes]
        l_m_mix = np.log10(sum(ave_gcounts_arr[:ngenes]))
        
        print('w_set = ',w_set, 'ngenes = ',ngenes,'\n')
        
        filename = 'w_set'+str(round(w_set,1))+'_ngenes'+str(ngenes) + '_npvecs'+str(npvecs)
        if output  is not None:
            log = open(os.path.join(output, filename+'.log.txt'),'w',buffering=1)
            log.write(filename+'\n')
        
        #print(2)
        w_list = []
        
        for i in range(nsamples):
            
            if output is None:
                print('doublet',i,'\n')
            else:
                log.write('doublet '+str(i)+'\n')
    
            
            # initialize 
            w_cur = w_init
            logR_m = np.log2(1/w_cur - 1)
            
            ll_dic, x, (p1_mat, p2_mat) = compare_samplingP_ll(ngenes, npvecs, alpha1, alpha2 , w_set, l_m_mix, return_ll=True, return_x_p=True, logR_m = logR_m, logR_std = 0.77)
            
            # hidx = np.argmax(ll_dic['logR'])
            # highlight_samplingP_ll(ll_dic,hidx)
        
            delta_w = ll_dic['w_x'] - w_cur
            w_cur = ll_dic['w_x']
            logR_m = np.log2(1/w_cur - 1)
            
            niter = 0
            one_way = True
            
            if np.abs(delta_w) < w_err:
                w_list.append(ll_dic['w_x'])
            else:
                while np.abs(delta_w) > w_err and one_way:

                    ll_dic_1 = compare_samplingP_ll(ngenes, npvecs, alpha1, alpha2, w_set, l_m_mix, return_ll=True, x_fix=x, real_ll_set=ll_dic['ll_real_set'], p1_fix = p1_mat, p2_fix = p2_mat,logR_m = logR_m, logR_std = 0.77)
                    
                    if output is None:
                        print('niter ', niter,', w = ', ll_dic_1['w_x'],'\n')
                    else:
                        log.write('niter '+str(niter)+', w = '+str(ll_dic_1['w_x'])+'\n')
                    
                    delta_w_old = copy.deepcopy(delta_w)
                    delta_w = ll_dic_1['w_x'] - w_cur
                    w_cur = ll_dic_1['w_x']
                    logR_m = np.log2(1/w_cur - 1)   
                    one_way = delta_w * delta_w_old > 0
                    niter += 1
                    
                w_list.append(ll_dic_1['w_x'])
        
        if output is not None:
            log.write('optimization completed.')
            log.close()
        
        
        logR_arr = np.log2(1/np.array(w_list)-1)
        w_fit = 1/(2**np.mean(logR_arr) + 1)
        
        normpdf = lambda rv,m,std:norm.pdf(rv, m, std)
        
        fig, axs = plt.subplots(1, 2, figsize=(8,3), dpi=128)
        
        plt.subplot(1,2,1)
        plt.hist(w_list, 50)
        plt.plot([w_set, w_set], [0,30], color='red')
        plt.plot([w_fit,w_fit],[0,30], color = 'yellow')
        #plt.show()
        plt.title('w', fontsize=14)
        
        plt.subplot(1,2,2)
        
        xx = np.linspace(logR_arr.min()-0.5, logR_arr.max()+0.5, 100)
        
        plt.hist(logR_arr, 50, density=True)
        plt.plot([np.log2(1/w_set-1),np.log2(1/w_set-1)],[0,0.6],color='red')
        plt.plot(xx, normpdf(xx, np.mean(logR_arr),np.std(logR_arr)), color='black')
        plt.plot([np.mean(logR_arr),np.mean(logR_arr)],[0,0.6],color='yellow')
        plt.title('logR', fontsize=14)
        
        tit = fig.suptitle('w_set = '+str(ll_dic['w_set'])+', w_fit = ' +str(round(w_fit, 3))+', '+str(ll_dic['ngenes'])+' genes, '+str(ll_dic['npvecs'])+' p pairs',fontsize=16)
        fig.subplots_adjust(top=0.85)
        fig.tight_layout()
        if output is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output, filename+'.jpg'),bbox_extra_artists=(tit,), bbox_inches='tight')
        #plt.show()
    
        return w_list
    
    except Exception as e:
        print(e)





get_cosine = lambda a: sum(a)/(np.linalg.norm(a)*np.sqrt(2))



def get_mg(raw_Alpha1, raw_Alpha2, merging_threshold=1, skip_threshold=2, alphaMin = 1):
    """
    fuse genes with alpha < 1 into metagenes in a balanced way

    Parameters
    ----------
    raw_Alpha1 : arr
        DESCRIPTION.
    raw_Alpha2 : arr
        DESCRIPTION.
    merging_threshold : float, optional
        DESCRIPTION. The default is 1.
    skip_threshold : float, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    mg_dic : TYPE
        DESCRIPTION.

    """
    #%% step 1: merge dims with tiny alpha to get metagenes (alpha > 1)
    
    raw_Alpha_arr = np.array([raw_Alpha1, raw_Alpha2])
    raw_cosine_vals = np.array([get_cosine(g) for g in raw_Alpha_arr.transpose()])
    
    # rank genes acording to the angle to (1,1)
    order_by_cosine = np.argsort(raw_cosine_vals)
    
    raw_set_1 = [g for g in order_by_cosine if raw_Alpha_arr[0,g] > raw_Alpha_arr[1,g]] 
    raw_set_2 = [g for g in order_by_cosine if raw_Alpha_arr[0,g] < raw_Alpha_arr[1,g]] 
    
    # extract dimensions with alpha > 1
    mg_single_set_1 = [g for g in raw_set_1 if raw_Alpha_arr[0,g] > alphaMin and raw_Alpha_arr[1,g] > alphaMin] 
    mg_single_set_2 = [g for g in raw_set_2 if raw_Alpha_arr[1,g] > alphaMin and raw_Alpha_arr[0,g] > alphaMin] 
    
    MetaGene_1_single = [[g] for g in mg_single_set_1]
    MetaGene_2_single = [[g] for g in mg_single_set_2]


    # set dimensions with alpha < 1 as seeds for merging meta genes
    mg_1_seeds = [g for g in raw_set_1 if g not in mg_single_set_1] 
    mg_2_seeds = [g for g in raw_set_2 if g not in mg_single_set_2] 
    
    # merge genes with smaller alpha in 1
    mg_1_merge = []
    while len(mg_1_seeds):
        
        # mg_1_seeds mean genes have larger alpah in 1 than 2
        # so merge genes in 1 until their alpha sum is greater than threshold in 2
        conterpart =  np.cumsum(raw_Alpha_arr[1, mg_1_seeds])
        
        idx = np.where(conterpart > merging_threshold)
        # idx = np.where(cumsum_1 > 1)
        if len(idx[0]) > 0:
            idx = idx[0][0]
            mg_1_merge.append(mg_1_seeds[:idx+1])
            del mg_1_seeds[:idx+1]
            
        else:
            # if left genes whose alpha sum < 1, add them into the last metagene
            if len(mg_1_merge) > 0:
                term = mg_1_merge.pop()
                term = term + mg_1_seeds
                mg_1_merge.append(term)
                # mg_1_seeds = []
            else:
                # if sum of all seed genes < merging_threshold, merge them with minimal single genes
                s_1_minidx = raw_Alpha1[mg_single_set_1].argmin()
                mg_1_merge = mg_1_seeds + MetaGene_1_single[s_1_minidx]
            
                del MetaGene_1_single[s_1_minidx], mg_single_set_1[s_1_minidx]
            
            break
        
    # merge genes with smaller alpha in 2
    mg_2_merge = []
    while len(mg_2_seeds):
        
        conterpart =  np.cumsum(raw_Alpha_arr[0, mg_2_seeds])
        
        idx = np.where(conterpart > merging_threshold)
        # idx = np.where(cumsum_2 > 1)
        if len(idx[0]) > 0:
            idx = idx[0][0]
            mg_2_merge.append(mg_2_seeds[:idx+1])
            del mg_2_seeds[:idx+1]
            
        else:
            # if left genes whose alpha sum < 1, add them into the last metagene
            if len(mg_2_merge) > 0:
                term = mg_2_merge.pop()
                term = term + mg_2_seeds
                mg_2_merge.append(term)
            else:
                # if sum of all seed genes < merging_threshold, merge them with minimal single genes
                s_2_minidx = raw_Alpha2[mg_single_set_2].argmin()
                mg_2_merge = mg_2_seeds + MetaGene_2_single[s_2_minidx]
            
                del MetaGene_2_single[s_2_minidx], mg_single_set_2[s_2_minidx]

            break
    
    
    if len(mg_1_merge) > 0:
        if isinstance(mg_1_merge[0],list):
            MetaGene_1 = mg_1_merge + MetaGene_1_single
        else:
            ### all genes with small alpha are merged int one
            MetaGene_1 = [mg_1_merge] + MetaGene_1_single
    else:
        MetaGene_1 = MetaGene_1_single
    
    
    if len(mg_2_merge) > 0:
        if isinstance(mg_2_merge[0],list):
            MetaGene_2 = mg_2_merge + MetaGene_2_single
        else:
            ### all genes with small alpha are merged int one
            MetaGene_2 = [mg_2_merge] + MetaGene_2_single        
    else:
        MetaGene_2 = MetaGene_2_single
    
    
    
    # metagenes, list of lists, recording the index of genes composing each meta gene 
    mg_list = MetaGene_1 + MetaGene_2
    
    
    #%% step 2: rank metagenes in probabilistically balanced way

    mg_Alpha_arr = np.array([[sum(raw_Alpha_arr[0,term]) for term in mg_list], 
                              [sum(raw_Alpha_arr[1,term]) for term in mg_list]])

    mg_cosine2XequalY = np.array([get_cosine(g) for g in mg_Alpha_arr.transpose()])
    
    mg_order_by_cosine = list( np.argsort(mg_cosine2XequalY) )
    
    mg_order_by_cosine.reverse()
    
    mg_idx_XgreaterY = [g for g in mg_order_by_cosine if mg_Alpha_arr[0,g] > mg_Alpha_arr[1,g]]
    mg_idx_YgreaterX = [g for g in mg_order_by_cosine if mg_Alpha_arr[0,g] <= mg_Alpha_arr[1,g]]

    
    mg_idx = [mg_idx_XgreaterY.pop(), mg_idx_YgreaterX.pop()]
    
    mg_Alpha_reordered = mg_Alpha_arr[:,mg_idx]
    
    cur_sum_x, cur_sum_y = mg_Alpha_reordered.sum(1)
    cur_std_x, cur_std_y = mg_Alpha_reordered.std(1)
    
    skip_idx_x = []
    skip_idx_y = []
    
    while len(mg_idx_XgreaterY)*len(mg_idx_YgreaterX):
        
        while cur_sum_x > cur_sum_y and len(mg_idx_YgreaterX) > 0:
            
            cur_mg_idx_y = mg_idx_YgreaterX.pop()
            cur_std_y_new = np.std(mg_Alpha_arr[1,mg_idx+[cur_mg_idx_y]])
            
            if cur_std_y_new/cur_std_y < skip_threshold:
            
                mg_idx.append(cur_mg_idx_y)
                mg_Alpha_reordered = mg_Alpha_arr[:,mg_idx]
                cur_sum_x, cur_sum_y = mg_Alpha_reordered.sum(1)
                cur_std_y = cur_std_y_new
            else:
                skip_idx_y.append(cur_mg_idx_y)
  
        while cur_sum_y > cur_sum_x and len(mg_idx_XgreaterY) > 0:
            
            cur_mg_idx_x = mg_idx_XgreaterY.pop()
            cur_std_x_new = np.std(mg_Alpha_arr[0,mg_idx+[cur_mg_idx_x]])
            
            if cur_std_x_new/cur_std_x < skip_threshold:
            
                mg_idx.append(cur_mg_idx_x)
                mg_Alpha_reordered = mg_Alpha_arr[:,mg_idx]
                cur_sum_x, cur_sum_y = mg_Alpha_reordered.sum(1)
                cur_std_x = cur_std_x_new
            else:
                skip_idx_y.append(cur_mg_idx_x)
  

    balanced_mg_num = len(mg_idx)
    
    mg_idx_all = mg_idx + skip_idx_y+ mg_idx_YgreaterX + skip_idx_x + mg_idx_XgreaterY
    mg_Alpha_reordered = mg_Alpha_arr[:,mg_idx_all]

    # list of lists, recording the index of genes composing each meta gene after reordering
    reordered_mg_pool = [mg_list[g] for g in mg_idx_all]
    #reordered_gidx_pool = [g for sub in reordered_mg_pool for g in sub]
    
    #mg_left = [mg_list[g] for g in range(len(mg_list)) if g not in mg_idx]
    #gidx_left = [g for sub in mg_left for g in sub]
    
    mg_dic = {'alpha1':mg_Alpha_reordered[0],
              'alpha2':mg_Alpha_reordered[1],
              'mg_pool':reordered_mg_pool,
              #'gidx_pool':reordered_gidx_pool,
              'balanced_mg_num':balanced_mg_num}
              #'mg_left':mg_left,
              #'gidx_left':gidx_left}
    
    return mg_dic






#%% debug

import seaborn as sns

def test_one_sample(Alpha1, Alpha2, w, w_mode = 'sampling', l_m_mix=2.1, l_std_mix=0.2,**para):
    """
    use one time of sampling to visualize doublet-generating p1-p2 pair, and w-estimating p1-pw pair.
    """
    posterior = para.get('posterior',False)
    gmean_1 = para.get('obs1',None)
    gmean_2 = para.get('obs2',None)
    
    p1sub = np.random.dirichlet(Alpha1, 1)[0,:]
    p2sub = np.random.dirichlet(Alpha2, 1)[0,:]
    
    w = 0.1
    p_wsum = p1sub*w + p2sub*(1-w)
    n = obtain_library_size(1, l_m_mix, l_std_mix)
    
    p_wsum[p_wsum==0] = 1e-10
    sum_term = p_wsum.sum()
    p_wsum = p_wsum/sum_term
    x = np.random.multinomial(n, p_wsum)
    
    if posterior:
        w_x_1p, p1_sampling, p2_sampling = estimate_w_with1p(x,Alpha1+gmean_1, Alpha2+gmean_2, mode=w_mode, return_p = True)
    else:
        w_x_1p, p1_sampling, p2_sampling = estimate_w_with1p(x,Alpha1, Alpha2, mode=w_mode, return_p = True)
    #w_x_mle_100p = estimate_w_mle(x, Alpha1, Alpha2, num_p = 100)
    
    cmap = sns.cubehelix_palette(as_cmap=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(8,7),dpi=128)
    
    ax = axs[0,0]
    pcm = ax.scatter(p1sub, p2sub, c=x, s=10, cmap=cmap)
    fig.colorbar(pcm, ax=ax)
    ax.set_title('P for generating X|p1*w+p2*(1-w)')
    ax.set(xlabel='P1', ylabel='P2')
    
    ax = axs[0,1]
    pcm = ax.scatter(p1_sampling, p2_sampling, c=x, s=10, cmap=cmap)
    fig.colorbar(pcm, ax=ax)
    ax.set_title('P for estimating w')
    ax.set(xlabel='P1', ylabel='P2')   
    
    ax = axs[1,0]
    pcm = ax.scatter(p1sub, p1_sampling, c=x, s=10, cmap=cmap)
    fig.colorbar(pcm, ax=ax)
    ax.set_title('Compare P1')
    ax.set(xlabel='P1 generating X', ylabel='P1 estimating w')      
    
    ax = axs[1,1]
    pcm = ax.scatter(p2sub, p2_sampling, c=x, s=10, cmap=cmap)
    fig.colorbar(pcm, ax=ax)
    ax.set_title('Compare P2')
    ax.set(xlabel='P2 generating X', ylabel='P2 estimating w')      

    fig.suptitle('w = '+str(w)+', w_1p = '+str(w_x_1p),fontsize=16)
    # fig.suptitle('w = '+str(w)+', w_1p = '+str(w_x_1p) +', w_mle100p = '+str(w_x_mle_100p),fontsize=16)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    plt.show()





def fit_w_Beta_mean(w_list,return_mean=False):
    w_tmp = np.array(copy.deepcopy(w_list))
    w_tmp[w_tmp==0] = 0.01
    w_tmp[w_tmp==1] = 0.99
    w_alpha, w_beta,_,_ = beta.fit(w_tmp, floc=0.,fscale=1.)
    
    if return_mean:
        return w_alpha/(w_alpha+w_beta)
    else:
        print('Mean of w: ', w_alpha/(w_alpha+w_beta))
        print('Mode of w:', (w_alpha-1)/(w_alpha+w_beta-2))



#%% previous gene ranking and ratio correction

def enum_all_genes(marker_seed, exp_vec_df, runname=''):   
    '''
    iterativly add a gene with minimal cosine to original ME genes  

    '''
    print(runname)
    # t_s = time.time()
    # cell1, cell2 = val.split('_')
    a_exp = exp_vec_df.iloc[:,0]  # exp_vec_df[cell1]
    b_exp = exp_vec_df.iloc[:,1]  # exp_vec_df[cell2]
    
    markers = copy.deepcopy(marker_seed)
    left_markers = [m for m in exp_vec_df.index if a_exp[m]*b_exp[m] > 0] # for ME genes, the product must be 0
    
    t_s = time.time()
    
    ab_sum = 0.0 # exp1_vec[markers]*exp2_vec[markers]
    a2_sum = np.sum(a_exp[markers]**2)
    b2_sum = np.sum(b_exp[markers]**2)
    
    cosine_curve = []
    
    while len(left_markers):
        if len(markers) % 200 == 0:
            print(runname,'cur markers',len(markers))
            
        #candidate_cosines = [(ab_sum + a_exp[m]*b_exp[m])/(np.sqrt(a2_sum + a_exp[m]**2)*np.sqrt(b2_sum + b_exp[m]**2)) 
        #                     for m in left_markers]
        candidate_cosines = [np.log(ab_sum+a_exp[m]*b_exp[m]) - 0.5*np.log(a2_sum + a_exp[m]**2) - 0.5*np.log(b2_sum + b_exp[m]**2) 
                             for m in left_markers]
         
        min_idx = np.argmin(candidate_cosines)
        best_marker = left_markers[min_idx]
        cosine_curve.append(candidate_cosines[min_idx])
        
        ab_sum += a_exp[best_marker]*b_exp[best_marker]
        a2_sum += a_exp[best_marker]**2
        b2_sum += b_exp[best_marker]**2
        
        markers.append(best_marker)
        left_markers.remove(best_marker)      
  
    print(runname, time.time()-t_s)
    
    return markers, cosine_curve




def ratio_correction(cur_ratio, cell1_rest_pmass, cell2_rest_pmass):
    """
    Correct mixing ratio with rest prabability mass.

    Parameters
    ----------
    cur_ratio : list, len: n_cells
        ratios estimated for each cells in cell_array.
    cell1_rest_pmass : float
        rest probability mass of cell type 1.
    cell2_rest_pmass : float
        rest probability mass of cell type 2.

    Returns
    -------
    float
        mixing ratio considering all gene.

    """
    numerator = (1 - cell2_rest_pmass) * cur_ratio
    denominator = 1 - cell1_rest_pmass + cur_ratio * cell1_rest_pmass - cur_ratio * cell2_rest_pmass
    return numerator / denominator
