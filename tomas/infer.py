#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 03:47:01 2022

@author: qy
"""
import time, os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm, dirichlet
import multiprocessing as mp
import pandas as pd
#import scanpy as sc


def ratio_2types(adata_dbl_mg,output,nrepeat=10):
    '''
    

    Parameters
    ----------
    adata_dbl_mg : AnnData
        The UMI count matrix of hetero-doublets in metagenes.
        Rows correspond to droplets and columns to metagenes.
    output : path
        Path to save the results.
    nrepeat : int, optional
        Times to repeat synthesizing doublets to infer ratios. The default is 10.

    Raises
    ------
    ValueError
        If 'output' gives invalid path, raise error.

    Returns
    -------
    esti_r_list : numpy.ndarray
        Estimate of total mRNA ratio of each hetero-dbouelt.

    '''
    if not os.path.exists(output):
        raise ValueError("Provide a valid path!")
    else:
        logpath = os.path.join(output,'rlog')
        if not os.path.exists(logpath):
            os.makedirs(logpath)
    
    
    mg_Alpha_new = adata_dbl_mg.varm['para_diri'] .transpose()
    Y_new = adata_dbl_mg.X
    
    estimateW(mg_Alpha_new,Y_new,logpath,nrepeat)
        
    ll_list = []
    for ridx in range(nrepeat):
        RandLL = pd.read_csv(os.path.join(logpath,'RandLL.'+str(ridx)+'.csv'),header=0,index_col=0)
        ll_list.append(RandLL['LL'].values[-1])
    
    ll_argmax = np.argmax(ll_list)
    #print(ll_argmax)

    ridx = ll_argmax
    
    R_est0 = pd.read_csv(os.path.join(logpath,'Rtrack.'+str(ridx)+'.csv'),header=0,index_col=0)
    esti_r_list = R_est0.iloc[:,-1].values
    
    #w_corrected = [ratio_correction(1/(r+1),left_mg_pmass[ct][0],left_mg_pmass[ct][1]) for r in esti_r_list]
    #r_list = [(1-w)/w for w in esti_r_list]
    
    return esti_r_list




def estimateW(mg_Alpha_new,Y_new,output,nrepeat=10):

    ndoublets = len(Y_new)

    Alpha_mat = mg_Alpha_new
    
    alpha0 = Alpha_mat[0]
    alpha1 = Alpha_mat[1]
    
    for repeat in range(nrepeat):
    
        log = open(os.path.join(output,'log.'+str(repeat)+'.txt'),'w',buffering=1)
    
        t0 = time.time() 
        log.write('start time '+str(t0)+'\n')
    
        # initialize 
        #w_init = initialize_w(Y_test, alpha0, alpha1)
        w_init = initialize_w(Y_new, alpha0, alpha1)
        r_init = (1-w_init)/w_init
        log.write('initizalized R '+str(r_init)+'\n')
        logR_m_record = []
        LL_record = []
        r_track = []
    
        logR_m = np.log2(r_init)
        logR_std = 1
        logR_m_record.append(logR_m)
    
        num_p = 1000
        num_w = 100
        maxiter = 50
        delta_LL_tol = 0.1
        job_unit = 20
        num_cores = 50
        parallel = True
    
        if ndoublets <= job_unit:
            parallel = False
    
        if parallel:
            # number of samples for each parallel job in E-step
            num_jobs = ndoublets // job_unit
            jobs_sidx = [i*job_unit for i in range(num_jobs)]
            jobs_sidx.append(ndoublets)
    
        keeptimes = []
        keeptimes_tol = 5
    
        LL = -1e308
        iteration = 0
        LL_record.append(LL)
    
        fp_dll0 = dirichlet.logpdf(alpha0/sum(alpha0), alpha0) 
        fp_dll1 = dirichlet.logpdf(alpha1/sum(alpha1), alpha1) 
    
        while( (len(keeptimes) <  keeptimes_tol) and (iteration < maxiter)):
            # Termination creteria: 
            # ------------------------- (1). iteration times exceed the maximum.
            # ------------------------- (2). the delta_LL is less than the threshold k times in a row.   
    
            # print('iteration',iteration)
            log.write('iteration '+str(iteration)+'\n')
            # input log, alpha0, alpha1, num_cores, 
    
            tp0 = time.time()
            if parallel:
    
                pool = mp.Pool(num_cores)  
                result_compact = [pool.apply_async( job_Estep, (Y_new, Alpha_mat, fp_dll0, fp_dll1, logR_m, logR_std,  num_p, num_w, jobs_sidx[i], jobs_sidx[i+1]) ) for i in range(num_jobs)]
                pool.close()
                pool.join()
    
                results = [term.get() for term in result_compact] 
                ll_mat = np.concatenate([term[0] for term in results])  # num_doublet x num_p
                w_mat = np.concatenate([term[1] for term in results])   # num_doublet x num_p
    
            else:
    
                ll_mat, w_mat = job_Estep(Y_new, Alpha_mat, fp_dll0, fp_dll1, logR_m, logR_std,  num_p, num_w, 0, len(Y_new))
    
            log.write('\tE step time:'+str(time.time()-tp0)+'\n')
    
            ### M-step:
            best_w_idx = [np.argmax(term) for term in ll_mat]
            ll_doublet = [ll_mat[idx][val] for idx,val in enumerate(best_w_idx)]
    
            delta_LL = np.sum(ll_doublet) - LL
    
            w_best = [w_mat[idx][val] for idx,val in enumerate(best_w_idx)]
            r_best = [(1-w)/w for w in w_best]
    
            if delta_LL > delta_LL_tol:
    
                LL = np.sum(ll_doublet)
                logR_m = np.mean(np.log2(r_best))
                logR_std = np.std(np.log2(r_best))
    
                logR_m_record.append(logR_m)
                LL_record.append(LL)
                r_track.append(r_best)

                log.write('\testi R '+str(2**logR_m)+'\n\tdelta LL '+str(delta_LL)+'\n')
    
            keeptimes.append(delta_LL < delta_LL_tol)
            if not np.all(keeptimes):
                keeptimes = []    
    
            iteration += 1
    
        tconsuming = time.time() - t0
    
        est_R = 2**logR_m_record[-1]#2**np.median(logR_m_record[2:]) if len(logR_m_record) > 2 else 2**logR_m_record[-1]

        log.write('final R '+str(est_R)+'\n')
        log.write('time '+str(tconsuming)+'\n\n\n')

        log.close()
    
        ## record R_estimation and LL each update
        R_LL_df = pd.DataFrame({'R':2**np.array(logR_m_record),'LL':LL_record})
        R_LL_df.to_csv(os.path.join(output,'RandLL.'+str(repeat)+'.csv'))
    
        ## record r estimated for each doublet per update
        r_track_df = pd.DataFrame(np.array(r_track).transpose(),columns=range(len(r_track)))
        r_track_df.to_csv(os.path.join(output,'Rtrack.'+str(repeat)+'.csv'))
    
    
    

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



def job_Estep(Y_test, Alpha_mat, fp_dll0, fp_dll1, logR_m, logR_std,  num_p, num_w, start_sidx, end_sidx):
    
    np.random.seed()
    
    sidx_range = range(start_sidx, end_sidx)
    alpha0, alpha1 = Alpha_mat
    
    ybatch_ll_list = []
    ybatch_w_list = []
  
    ### E-setp:
    for i in sidx_range:
        
        y = Y_test[i]
        
        logR_sampling = np.random.normal(logR_m, logR_std, num_w)
        W = 1/(2**logR_sampling+1)
    
        logR_ll = norm.logpdf(logR_sampling, logR_m, logR_std) 
        
        d_p0_sampling  = np.random.dirichlet(alpha0*100, num_p)
        d_p0_sampling = get_P_valid(d_p0_sampling)
        diri_ll_0 = [dirichlet.logpdf(p, alpha0)  - fp_dll0 for p in d_p0_sampling]
        
        d_p1_sampling  = np.random.dirichlet(alpha1*100, num_p)
        d_p1_sampling = get_P_valid(d_p1_sampling)
        diri_ll_1 = [dirichlet.logpdf(p, alpha1) - fp_dll1 for p in d_p1_sampling]
                
        y_ll = []
        for w_idx in range(len(W)):
            
            w = W[w_idx]
            p_wsum = d_p0_sampling * w + d_p1_sampling * (1-w)
 
            sum_term = p_wsum.sum(1)
            p_wsum = p_wsum/sum_term.reshape(-1,1)       
            
            mult_ll = (np.log(p_wsum)*y).sum(1)

            pvec_y_ll = np.array(diri_ll_0)+np.array(diri_ll_1)+logR_ll[w_idx]+np.array(mult_ll) 
            #pvel_y_ll_midx = np.argmax(pvec_y_ll)
    
            y_ll.append(np.max(pvec_y_ll))

        ybatch_ll_list.append(y_ll)
        ybatch_w_list.append(W) 
        
  
    ll_mat = np.array(ybatch_ll_list) # ndoublet x num_w
    w_mat = np.array(ybatch_w_list)   # ndoublet x num_w
    
    return ll_mat, w_mat



def get_P_valid(P):
    
    P_valid = np.zeros(P.shape)
    for i in range(P.shape[0]):
        p_tmp = P[i]
        p_tmp[p_tmp == 0] = 1e-10
        p_tmp_sum = sum(p_tmp)
        p_tmp = p_tmp/p_tmp_sum
        
        P_valid[i,:] = p_tmp
        
    return P_valid

