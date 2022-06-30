#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:13:43 2021

@author: qy
"""

import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from scipy.stats import norm
import copy
import os
import numpy as np
from auxi import get_P_valid, sweep_w_maxLL
import pickle
import pandas as pd
from scipy.stats import gaussian_kde
import time




from auxi import get_mg

def obtain_mg(raw_Alpha1_0, raw_Alpha2_0, kl, kl_cutoff=1, merging_threshold=5, skip_threshold=2,alphaMin = 1):
    
    # use genes with KL greater than cutoff to obtain meta-genes
    KL_filter_idx = np.where(kl > kl_cutoff)[0]
    
    raw_Alpha1 = raw_Alpha1_0[KL_filter_idx]
    raw_Alpha2 = raw_Alpha2_0[KL_filter_idx]

    mg_dic = get_mg(raw_Alpha1, raw_Alpha2, merging_threshold, skip_threshold,alphaMin)
    
    mg_pool = mg_dic['mg_pool'] 
    #balanced_mg_num = mg_dic['balanced_mg_num']
    
    # original idx of genes used to generate metagenes, each element is a list of gene idx
    mg_dic['mg_gidx'] = [list(KL_filter_idx[term]) for term in mg_pool]
    mg_dic['mg_genepool'] = [g for sub in  mg_dic['mg_gidx']  for g in sub]
    
    return mg_dic




def initialize_w(Y, alpha1, alpha2):
    
    #plot = plot_
    
    p1 = alpha1/sum(alpha1)
    p2 = alpha2/sum(alpha2)
    gsum_mix = Y.sum(0)
    pmix = gsum_mix/sum(gsum_mix)
    idx_in = [i for i in range(len(p1)) if pmix[i] > min(p1[i],p2[i]) and pmix[i] < max(p1[i], p2[i])]
    
    w_tmp = [(pmix[i]-p2[i])/(p1[i]-p2[i]) for i in idx_in if p1[i] - p2[i] != 0]
    
    kde = gaussian_kde(w_tmp)
    samples = np.linspace(0, 1, 1000)
    probs = kde.evaluate(samples)
    maxima_index = probs.argmax()
    maxima = samples[maxima_index]

    return maxima





def estimate_w_mle(y, Alpha1, Alpha2, num_p = 1000, **para):
    
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
        
        w_tmp, ll, mult_ll, diri_ll_1, diri_ll_2, logR_ll = sweep_w_maxLL(y, Alpha1, Alpha2, p1_sampling, p2_sampling, logR_m, logR_std)
        
        d1_list.append(diri_ll_1)
        d2_list.append(diri_ll_2)
        w_list.append(w_tmp)
        mu_list.append(mult_ll)
        ll_list.append(ll)
        logR_list.append(logR_ll)
        

    return w_list[np.argmax(ll_list)], np.array([ll_list, w_list, d1_list, d2_list, mu_list, logR_list])





def deconvolute_y(y, num_p, alpha1, alpha2, **para):
        
    w_est, ll_w_arr = estimate_w_mle(y, alpha1, alpha2, num_p = num_p, **para)
    ll_max = [np.max(ll_w_arr[0,:nidx]) for nidx in range(1,num_p+1)]
    
    ll_dic = {'alpha1':alpha1,
              'alpha2':alpha2,
              'diri1': ll_w_arr[2], 
              'diri2': ll_w_arr[3], 
              'mult': ll_w_arr[4], 
              'logR': ll_w_arr[5], 
              'completell': ll_w_arr[0],
              'accumulatedMaxll': ll_max,
              'w':ll_w_arr[1],
              #'ll_real_complete':ll_real_complete,
              #'ll_real_set':ll_real_set,
              #'w_real': w_real,              
              'w_est':w_est,
              'ngenes': len(y),
              'npvecs': num_p}    

    return ll_dic



def deconvolute_Y(Alpha_arr, Y, ngenes, num_p, w_init, w_err=0.001, output=None, **para):
    
    w_real = para.get('w_real', None) # para for simulation
    r_real = para.get('r_real', None)
    
    if w_real is not None:
        r_real = (1-w_real)/w_real
        filename = 'Simulation_w'+str(round(w_real,1))+'_ngenes'+str(ngenes) + '_npvecs'+str(num_p)
    elif r_real is not None:
        w_real = 1/(1+r_real)
        filename = 'Simulation_r'+str(round(r_real,1))+'_ngenes'+str(ngenes) + '_npvecs'+str(num_p)
    else:
        filename = 'RealData'+'_ngenes'+str(ngenes) + '_npvecs'+str(num_p)
   
    if output  is not None:
        log = open(os.path.join(output, filename+'.log.txt'),'w',buffering=1)
        log.write(filename+'\n')
    else:
        print(filename,'\n')
    
    
    #w_init = initialize_w(Y, alpha1, alpha2)
    nsamples = len(Y)
    
    alpha1 = Alpha_arr[0,:ngenes]
    alpha2 = Alpha_arr[1,:ngenes]
    Y = Y[:,:ngenes]

    # optimize w for each doublet
    t0 = time.time()
    w_list = []
    for sidx in range(nsamples):
        
        y = Y[sidx]
        
        if output is None:
            print('doublet', sidx,'\n')
        else:
            log.write('doublet '+str(sidx)+'\n')
            
        p1_sampling_mat  = np.random.dirichlet(alpha1, num_p)
        p2_sampling_mat  = np.random.dirichlet(alpha2, num_p)
        
        p1_sampling_mat = get_P_valid(p1_sampling_mat)
        p2_sampling_mat = get_P_valid(p2_sampling_mat)
    
        w_cur = copy.deepcopy(w_init)
        logR_m = np.log2(1/w_cur - 1)
    
        niter = 0
        ll_dic = deconvolute_y(y, num_p, alpha1, alpha2, p1_fix = p1_sampling_mat, p2_fix = p2_sampling_mat,logR_m = logR_m, logR_std = 0.77)
        if output is None:
            print('niter ', niter,', w = ', ll_dic['w_est'],'\n')
        else:
            log.write('niter '+str(niter)+', w = '+str(ll_dic['w_est'])+'\n')
            
        delta_w = w_cur - ll_dic['w_est']
        one_way = True
        niter += 1
        
        while np.abs(delta_w) > w_err and one_way:

            ll_dic = deconvolute_y(y, num_p, alpha1, alpha2, p1_fix = p1_sampling_mat, p2_fix = p2_sampling_mat,logR_m = logR_m, logR_std = 0.77)
            
            if output is None:
                print('niter ', niter,', w = ', ll_dic['w_est'],'\n')
            else:
                log.write('niter '+str(niter)+', w = '+str(ll_dic['w_est'])+'\n')
            
            delta_w_old = copy.deepcopy(delta_w)
            delta_w = ll_dic['w_est'] - w_cur
            w_cur = ll_dic['w_est']
            logR_m = np.log2(1/w_cur - 1)   
            one_way = delta_w * delta_w_old > 0
            niter += 1
            
        w_list.append(ll_dic['w_est'])

    if output is not None:
        log.write('optimization completed, taking ' + str(time.time()-t0) +' seconds.\n')
        log.close()
    
    # visualize w and logR
    logR_arr = np.log2(1/np.array(w_list)-1)
    w_fit = 1/(2**np.mean(logR_arr) + 1)
    
    normpdf = lambda rv,m,std:norm.pdf(rv, m, std)
    
    fig, axs = plt.subplots(1, 2, figsize=(8,3), dpi=128)
    
    plt.subplot(1,2,1)
    (cnt, _, _) = plt.hist(w_list, 50)
    if w_real is not None:
        plt.plot([w_real, w_real], [0,np.max(cnt)*1.2], color='red')
    plt.plot([w_fit,w_fit],[0,np.max(cnt)*1.2], color = 'yellow')
    #plt.show()
    plt.title('w', fontsize=14)
    plt.legend()
    
    plt.subplot(1,2,2)
    
    xx = np.linspace(logR_arr.min()-0.5, logR_arr.max()+0.5, 100)
    
    (cnt, _, _) = plt.hist(logR_arr, 50, density=True)
    if w_real is not None:
        plt.plot([np.log2(1/w_real-1),np.log2(1/w_real-1)],[0,np.max(cnt)*1.2],color='red',label='true R')
    plt.plot(xx, normpdf(xx, np.mean(logR_arr),np.std(logR_arr)), color='black',label='fit Gaussian')
    plt.plot([np.mean(logR_arr),np.mean(logR_arr)],[0,np.max(cnt)*1.2],color='yellow',label='mean R')
    plt.title('logR', fontsize=14)
    plt.legend()
    
    setting = 'realdata' if r_real is None else str(r_real)
    tit = fig.suptitle('setting: '+setting+', r_fit = ' +str(round(2**np.mean(logR_arr) , 3))+', '+str(ll_dic['ngenes'])+' genes, '+str(ll_dic['npvecs'])+' p pairs',fontsize=16)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output, filename+'.jpg'),bbox_extra_artists=(tit,), bbox_inches='tight')
    #plt.show()

    return w_list, logR_arr, w_fit





Diri_ll = lambda alpha, p: sum((alpha-1)*np.log(p))
get_w_maxll = lambda w_arr, ll_arr: w_arr[np.argmax(ll_arr)]

import multiprocessing as mp


def job_Estep(Y_test, Alpha_mat, logR_m, logR_std, fp_dll1, fp_dll2, num_p, num_w, start_sidx, end_sidx):
    
          
    #num_p = len(alpha1)
    sidx_range = range(start_sidx, end_sidx)
    
    fp_logr = norm.logpdf(logR_m, logR_m, logR_std)
    
    ybatch_ll_list = []
    ybatch_w_list = []
    
    for sidx in sidx_range:

        y = Y_test[sidx]
        
        logR_sampling = np.random.normal(logR_m, logR_std, num_w)
        W = 1/(2**logR_sampling+1)
        
        logR_ll = norm.logpdf(logR_sampling, logR_m, logR_std) - fp_logr
                  
        p1_sampling_mat  = np.random.dirichlet(Alpha_mat[0], num_p)
        p2_sampling_mat  = np.random.dirichlet(Alpha_mat[1], num_p)   
        
        p1_sampling_mat = get_P_valid(p1_sampling_mat)
        p2_sampling_mat = get_P_valid(p2_sampling_mat)
        
        diri_ll_1 = [Diri_ll(Alpha_mat[0], p) - fp_dll1 for p in p1_sampling_mat]
        diri_ll_2 = [Diri_ll(Alpha_mat[1], p) - fp_dll2 for p in p2_sampling_mat]
                
        y_ll = []
        for w_idx in range(len(W)):
            w = W[w_idx]
            p_wsum = p1_sampling_mat*w + p2_sampling_mat*(1-w)
            
            sum_term = p_wsum.sum(1)
            p_wsum = p_wsum/sum_term.reshape(-1,1)    

            mult_ll = (np.log(p_wsum)*y).sum(1)
            
            y_w_ppair_ll = np.array(diri_ll_1)+np.array(diri_ll_2)+logR_ll[w_idx]+np.array(mult_ll)
            fp_mult = np.max(y_w_ppair_ll)
            
            P_y_w = sum(np.exp(y_w_ppair_ll-fp_mult))
            #P_y_w = np.max(np.exp(y_w_ppair_ll-fp_mult))
            
            if P_y_w == 0:
                y_ll.append(1e-323)
            elif P_y_w == np.inf:
                y_ll.append(1e308)
            else:
                y_ll.append(np.log(P_y_w)+fp_mult)
        
        ybatch_ll_list.append(y_ll)
        ybatch_w_list.append(W)
        
    return ybatch_ll_list, ybatch_w_list



def optimize_logR_EM(Y_test, Alpha_mat, r_init=None,**para):

    alpha1 = Alpha_mat[0]
    alpha2 = Alpha_mat[1]
    
    # initialization 
    w_init = para.get('w_init', None)
    log = para.get('log',None)
    parallel = para.get('parallel',True)
    num_cores = para.get('num_cores',int(mp.cpu_count()*0.9))

    if w_init is None and r_init is None:
        print('Error: No necessary initialization parameters. please set w_init or r_init.')
        return
    elif w_init is not None:
        logR_m = np.log2(1/w_init - 1)
    else:
        logR_m = np.log2(r_init)
    
    if log is None:
        print('Optimize logR via EM. r initialized as ', 2**logR_m)
    else:
        log.write('Optimize logR via EM. r initialized as ' + str(2**logR_m) + '\n')
        
    t0 = time.time()
    #w_cur = copy.deepcopy(w_init)
    #logR_m = np.log2(1/w_cur - 1) # 4 # np.log2(1) # np.log2(1) #
    logR_std = para.get('logR_std',1)

    num_p = para.get('num_p',1000)
    num_w = para.get('num_w',100)
    maxiter = para.get('maxiter',100)
    delta_ll_tol = para.get('delta_ll',0.1)
    
    keeptimes = []
    keeptimes_tol = para.get('keeptimes_tol',5)
    
    ll_sum = -1e308
    iteration = 0
    num_sample = len(Y_test)
    
    job_unit = para.get('job_unit',100)
    
    if parallel:
         # number of samples for each parallel job in E-step
        num_jobs = num_sample // job_unit
        jobs_sidx = [i*job_unit for i in range(num_jobs)]
        jobs_sidx.append(num_sample)

    if num_sample <= job_unit:
        parallel = False
    
    
    fp_dll1 = Diri_ll(alpha1, alpha1/sum(alpha1))
    fp_dll2 = Diri_ll(alpha2, alpha2/sum(alpha2))  
    
    while( (len(keeptimes) <  keeptimes_tol) and (iteration < maxiter)):

        #logR_sampling = np.random.normal(logR_m, logR_std, num_w)
        #W = 1/(2**logR_sampling+1)
        
        #W = np.linspace(0.01,1-0.01,99)
        #logR_sampling = (1-W)/W
        #num_w = len(logR_sampling)
         
        fp_logr = norm.logpdf(logR_m, logR_m, logR_std)
        #logR_ll = norm.logpdf(logR_sampling, logR_m, logR_std) - fp_logr
        
        ll_track = []
        
        ll_doublet = []
        w_doublet = []
        #weight_doublet = []
        #Lambda_y_w = np.zeros((num_sample, num_w))
        
        ### E-step:
        if parallel:
            t0 = time.time()
            pool = mp.Pool(num_cores)  
            result_compact =[pool.apply_async( job_Estep, (Y_test, Alpha_mat, logR_m, logR_std, fp_dll1, fp_dll2, num_p, num_w, jobs_sidx[i], jobs_sidx[i+1]) ) for i in range(num_jobs)]
            pool.close()
            pool.join()

            results = [term.get() for term in result_compact]
            results_ll = [term[0] for term in results]
            results_w = [term[1] for term in results]
            y_w_ll_list = [val for sub in results_ll for val in sub]
            y_w_list = [val for sub in results_w for val in sub]
            
        else:
            t0 = time.time()
            y_w_ll_list, y_w_list = job_Estep(Y_test, Alpha_mat, logR_m, logR_std, fp_dll1, fp_dll2, num_p, num_w, 0, num_sample)
            
        if log is None:
            print('\ttime: ', time.time()-t0)
        else:
            log.write('\ttime: '+str(time.time()-t0)+'\n')

            
        ### M-step I:
        y_w_idx = [np.argmax(y_w_ll) for y_w_ll in y_w_ll_list] #np.argmax(y_w_ll)
        w_doublet = [y_w_list[idx][val] for idx,val in enumerate(y_w_idx)]
        ll_doublet = [np.max(y_w_ll) + fp_logr for y_w_ll in y_w_ll_list]
   
        delta_ll = sum(ll_doublet) - ll_sum
        '''
        if log is None:
            print('niter =',iteration, ', delta_ll:',delta_ll)
        else:
            log.write('niter = '+str(iteration)+', delta_ll: '+str(delta_ll)+'\n')
        '''
        ### M-step II:
        if delta_ll > delta_ll_tol:
            
            ll_sum = sum(ll_doublet)
            ll_track.append(ll_track)
        
            logR_maxll = [np.log2(1/w - 1) for w in w_doublet]
            #logR_m = sum( np.array(logR_maxll) * np.array(weight_doublet) )/sum(weight_doublet)
            #logR_std = sum( (np.array(logR_maxll) - logR_m)**2 * np.array(weight_doublet) )/sum(weight_doublet)
            
            logR_m  = np.mean(logR_maxll)
            logR_std = np.std(logR_maxll)
            
            if logR_std == 0:
                break
            '''
            if log is None:
                print('\tparameter updated: 2**logR_m = ', 2**logR_m, ', logR_std = ', logR_std)
            else:
                log.write('\tparameter updated: 2**logR_m = '+str(2**logR_m)+', logR_std = '+str(logR_std)+'\n')
            '''
        keeptimes.append(delta_ll < delta_ll_tol)
        if not np.all(keeptimes):
            keeptimes = []
    
        iteration += 1
    
    if log is None:
        print('optimization completed. 2**logR_m = ', 2**logR_m, ', logR_std = ', logR_std,', ll =', ll_sum)
        print('total time:', time.time() - t0)
    else:
        log.write('optimization completed. 2**logR_m = '+str(2**logR_m)+', logR_std = '+str(logR_std)+', ll ='+str(ll_sum)+'\n')
        log.write('total time: '+str(time.time() - t0)+'\n\n\n')
        
    return ll_sum, logR_maxll, (logR_m, logR_std)
    






from scipy.special import loggamma, gammaln, digamma


def KL_Dir_miginal(alpha1, alpha2, gidx):
    
    # gammaln(1e-309) = inf, gammaln(1e-308) = 709.1962086421661
    alpha1[alpha1 < 1e-6] = 1e-6
    alpha2[alpha2 < 1e-6] = 1e-6
    
    #print(gidx)
    g_alpha1 = alpha1[gidx]
    g_alpha2 = alpha2[gidx]
    left_alpha1 = sum(alpha1) - g_alpha1
    left_alpha2 = sum(alpha2) - g_alpha2
    kl = gammaln(sum(alpha1)) - gammaln(g_alpha1) - gammaln(left_alpha1) - gammaln(sum(alpha2)) + gammaln(g_alpha2) + gammaln(left_alpha2) + \
         (g_alpha1 - g_alpha2)*(digamma(g_alpha1)-digamma(sum(alpha1))) + (left_alpha1 - left_alpha2)*(digamma(left_alpha1) - digamma(sum(alpha1)))

    return kl


