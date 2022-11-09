#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 02:41:49 2022

@author: qy
"""




#import pickle
import numpy as np
import scipy
from scipy.special import gammaln, digamma
import os
import pandas as pd
import scanpy as sc
import tqdm
import multiprocessing as mp
import math
from scipy import stats




#%% meta-genes

def cal_KL_bc(adata, groups, parallel=True):
    '''
    Calculate the KL-divergence of genes betweeo two Dirichlet-Multinomial distribution.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    groups : list of strings
        Two droplet categories, e.g. ['Homo-ct1', 'Homo-ct2'] annotated in adata.obs[groupby] to which KL-divergence should be calculated with.
    parallel : bool, optional
        If or not to run in parallel. The default is True.

    Returns
    -------
    None.

    '''

    raw_Alpha_1 = adata.varm['para_diri'][groups[0]]
    raw_Alpha_2 = adata.varm['para_diri'][groups[1]]
    
    if parallel:
        
        tot_objs = len(raw_Alpha_1)
        
        num_cores = int(mp.cpu_count()*0.8)
        job_unit = math.floor(tot_objs/num_cores)
        
        num_jobs = tot_objs // job_unit
        jobs_sidx = [i*job_unit for i in range(num_jobs)]
        jobs_sidx.append(tot_objs)
        
        pool = mp.Pool(num_cores)  
        result_compact = [pool.apply_async( job_KL, (raw_Alpha_1,raw_Alpha_2, jobs_sidx[i], jobs_sidx[i+1]) ) for i in range(num_jobs)]
        pool.close()
        pool.join()        
        
        kl = np.array([v for term in result_compact for v in term.get()])
        
    else:
        
        # KL-metric for each gene
        kl_1_2 = []
        kl_2_1 = []
        
        # add progression bar if not multiprocess
        for gidx in tqdm.tqdm(range(len(raw_Alpha_1))):
            
            kl_tmp = KL_Dir_miginal(raw_Alpha_1, raw_Alpha_2, gidx) 
            kl_1_2.append(kl_tmp)
            
            kl_tmp = KL_Dir_miginal(raw_Alpha_2, raw_Alpha_1, gidx) 
            kl_2_1.append(kl_tmp)
        
        kl = (np.array(kl_1_2)+np.array(kl_2_1))/2
    
    '''
    info = {'raw_alpha1':raw_Alpha_1,
            'raw_alpha2':raw_Alpha_2,
            'kl':kl}
    
    f = open(os.path.join(output,'dmn.kl.pickle'),'wb')
    pickle.dump(info,f)
    f.close()
    '''
    #kl = obtain_KL_given2alphaVec(raw_Alpha_1, raw_Alpha_2, output)
    adata.varm['kl'] = pd.DataFrame({groups[0]+'_'+groups[1]:kl},index=adata.var_names)    
    
    #return kl
    
    



def get_dbl_mg_bc(adata, groupby, groups, save_path=None, kl_filter=None, num_mg = 100, kl_cutoff=1, merging_threshold=5, skip_threshold=2,alphaMin = 1):
    '''
    Extract metagenes from raw UMI counts of heterotypic doublets.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    groups : list of strings
        Specify two droplet categories, e.g. ['Homo-ct1', 'Homo-ct2'] annotated in adata.obs[groupby], which form the hetero-dbl of interest.
    save_path : path
        Path to save the results.
    num_mg : int, optional
        Number of exclusive meta-genes. The default is 100.
    kl_cutoff : float, optional
        DESCRIPTION. The default is 1.
    merging_threshold : float, optional
        Stop merging when alpha sum of metagenes exceeds the threshold in conterpart celltype. The default is 5.
    skip_threshold : float, optional
         If adding a gene into current metegene leads to 'skip_threshold'-fold change of the std of alpha values,
         this gene is skipped. The default is 2.
    alphaMin : float, optional
         Individual genes with alpha greater than the threshold are skipped in mergging step and considerted as a specifial metagene with one member gene. The default is 1.

    Returns
    -------
    adata_dbl_mg_top : TYPE
        DESCRIPTION.

    '''
    raw_Alpha1_0 = adata.varm['para_diri'][groups[0]]
    raw_Alpha2_0 = adata.varm['para_diri'][groups[1]]
    
    raw_Alpha1_0[raw_Alpha1_0 < 1e-6] = 1e-6
    raw_Alpha2_0[raw_Alpha2_0 < 1e-6] = 1e-6
    
    #kl = obtain_KL_given2alphaVec(raw_Alpha_1, raw_Alpha_2, output)
    #adata.varm['kl'] = pd.DataFrame({groups[0]+'_'+groups[1]:kl},index=adata.var_names)
    
    if kl_filter:
        # use genes with KL greater than cutoff to obtain meta-genes
        kl = adata.varm['kl']
        KL_filter_idx = np.where(kl > kl_cutoff)[0]
    else:
        KL_filter_idx = np.array([i for i in range(len(raw_Alpha1_0))])
        
    raw_Alpha1 = raw_Alpha1_0[KL_filter_idx]
    raw_Alpha2 = raw_Alpha2_0[KL_filter_idx]

    mg_dic = get_mg(raw_Alpha1, raw_Alpha2, merging_threshold, skip_threshold,alphaMin)
    
    mg_pool = mg_dic['mg_pool'] 
    #balanced_mg_num = mg_dic['balanced_mg_num']
    
    # original idx of genes used to generate metagenes, each element is a list of gene idx
    #mg_dic['mg_gidx'] = [list(KL_filter_idx[term]) for term in mg_pool]
    #mg_dic['mg_genepool'] = [g for sub in  mg_dic['mg_gidx']  for g in sub]
    #mg_gidx = mg_dic['mg_gidx']
    #mg_genepool = mg_dic['mg_genepool']
    
    mg_gidx = [list(KL_filter_idx[term]) for term in mg_pool]
    #mg_genepool = [g for sub in  mg_dic['mg_gidx']  for g in sub]
    mg_genepool = [g for sub in  mg_gidx  for g in sub]
    
    mg_alpha1 = mg_dic['alpha1']
    mg_alpha2 = mg_dic['alpha2']
    
    left_gidx = list(set(range(len(raw_Alpha1_0))).difference(mg_genepool)) 
    mg_gidx.append(left_gidx)
    

    # merge alpha and counts according to meta-genes
    count_dbl = adata[adata.obs[groupby]==groups[2],:].X
    if isinstance(count_dbl, scipy.sparse.csr.csr_matrix):
        count_dbl = count_dbl.toarray()
        
    mg_Alpha_arr = np.concatenate((np.array([mg_alpha1, mg_alpha2]), np.array([[sum(raw_Alpha1_0)-sum(mg_alpha1)],[sum(raw_Alpha2_0)-sum(mg_alpha2)]])),axis=1)
    Y = np.transpose(np.array([np.sum(count_dbl[:,term],1) for term in mg_gidx]))
    
    adata_dbl_mg = sc.AnnData(Y)
    adata_dbl_mg.obs_names = adata.obs_names[adata.obs[groupby]==groups[2]]
    adata_dbl_mg.varm['para_diri'] = mg_Alpha_arr.transpose()
    #adata_dbl_mg.uns['mg_gidx'] = mg_gidx#,index=adata_dbl_mg.var_names)
    if save_path is not None:
        if os.path.exists(save_path):
            adata_dbl_mg.write_h5ad(os.path.join(save_path,groups[0]+'_'+groups[1]+'_dbl_mg.h5ad'))
        else:
            raise ValueError("'save_path' is invalid. please input a valid path")
        
    # extract top num_mg 
    topmg_idx = [[i] for i in range(num_mg)] + [list(range(num_mg,  len(mg_gidx)))]
    mg_Alpha_new = np.transpose(np.array([np.sum(mg_Alpha_arr[:,term],1) for term in topmg_idx]))
    Y_new = np.transpose(np.array([np.sum(Y[:,term],1) for term in topmg_idx]))
    
    adata_dbl_mg_top = sc.AnnData(Y_new)
    adata_dbl_mg_top.obs_names = adata.obs_names[adata.obs[groupby]==groups[2]]
    adata_dbl_mg_top.varm['para_diri'] = mg_Alpha_new.transpose()
    #adata_dbl_mg_top.uns['mg_gidx'] = mg_gidx[:num_mg] + [[item  for idx,val in enumerate(mg_gidx) for item in val]]#,index=adata_dbl_mg_top.var_names)
    
    return adata_dbl_mg_top        





'''
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
'''




def job_KL(raw_Alpha_1,raw_Alpha_2,start_sidx, end_sidx):
    
    sub_kl = []
    for gidx in range(start_sidx,end_sidx):

        kl_1_2 = KL_Dir_miginal(raw_Alpha_1, raw_Alpha_2, gidx) 
        kl_2_1 = KL_Dir_miginal(raw_Alpha_2, raw_Alpha_1, gidx) 
        
        kl = (kl_1_2+kl_2_1)/2
        sub_kl.append(kl)
        
    return sub_kl



    
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
    ## step 1: merge dims with tiny alpha to get metagenes (alpha > 1)
    
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
    
    
    ## step 2: rank metagenes in probabilistically balanced way

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




#%% correct UMI 

import copy

def correctUMI(adata, groupby, ratios, method='upsampling', logUMIby=None):
    '''
    Correct UMI to meet the estimated total-mRNA-ratio.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    ratios : dict
        DESCRIPTION.
    logUMIby : str, optional
        The key of total UMIs in log10 stored in adata.obs. The default is None.
        
    Returns
    -------
    adata_rc : TYPE
        The corrected UMI count matrix.

    '''    

    groups = list(ratios.keys())
    r_vals = list(ratios.values())
    resortidx = np.argsort(r_vals)
    
    groups = [groups[v] for v in resortidx]
    r_vals = [r_vals[v] for v in resortidx]

    alpha_df = adata.varm['para_diri']

    if logUMIby is None:

        adata.obs['total_UMIs'] = adata.X.sum(1)[:,0]
        adata.obs['log10_totUMIs'] = np.log10(adata.obs['total_UMIs'])
        logUMIby = 'log10_totUMIs'
        
    #logN_para = [norm.fit(adata.obs[logUMIby][adata.obs[groupby]==v]) for v in groups]
    #mu_list = [x[0] for x in logN_para]
    #std_list = [x[1] for x in logN_para]
    mu_list = list(adata.uns['logUMI_para'].loc[groups,'mean'])
    
    mat_rc = copy.deepcopy(adata.X.toarray())
    
    for i in range(len(groups)-1):
        
        print('Correct UMIs of population '+groups[i+1])

        UMI_delta = 10**(mu_list[0] + np.log10(r_vals[i+1]) - mu_list[i+1])-1
        alpha = alpha_df[groups[i+1]]
        X_mat = copy.deepcopy(mat_rc[adata.obs[groupby]==groups[i+1],:])
        
        X_mat_up = _correct(X_mat, alpha, UMI_delta, method)
        mat_rc[adata.obs[groupby]==groups[i+1],:] = X_mat_up
    
    adata_rc = sc.AnnData(mat_rc)
    #adata_rc.obs = adata.obs
    adata_rc.var_names = adata.var_names.values.tolist()
    adata_rc.obs_names = adata.obs_names.values.tolist()
    adata_rc.uns['corrected'] = 'data'

    return adata_rc



def _correct(X_mat, alpha, UMI_delta, method):

    if isinstance(X_mat, scipy.sparse.csr.csr_matrix):
        X_mat = X_mat.toarray()
    
    if method == 'upscaling':
        
        X_mat_up = X_mat * (1+UMI_delta)
        X_mat_up = X_mat_up.astype(int)
        
    elif method == 'upsampling':
        x_list = []
        for cidx in tqdm.tqdm(range(X_mat.shape[0])):
            x = X_mat[cidx]
            p = np.random.dirichlet(alpha+x,size=1)
            x_delta = np.random.multinomial(int(sum(x)*UMI_delta), p[0])
            x_modified = x + x_delta
            x_list.append(x_modified)
    
        X_mat_up = np.array(x_list)
        #X_mat_up_sparse = scipy.sparse.csr_matrix(X_mat_up)
    else:
        raise ValueError("'method' value error: only 'upsampling' or 'upscaling' is supported.")

    return X_mat_up #_sparse




def rm_outliers(x):
    iqr = stats.iqr(x)
    outlier_lb = np.quantile(x,0.25)-1.5*iqr
    outlier_ub = np.quantile(x,0.75)+1.5*iqr
    x_shrinkage = x[x > outlier_lb]
    x_shrinkage = x_shrinkage[x_shrinkage<outlier_ub]
    return x_shrinkage#,(outlier_lb,outlier_ub)



from scipy.stats import norm
 
def correct_para(adata_sgl, groupby, achor_group=None, adjust_group=None):
    
    mean_logR, std_logR = norm.fit(rm_outliers(np.log10(adata_sgl.uns['Rest_perdbl'])))
    
    adata_sgl.uns['raw_logUMI_para'] = copy.deepcopy(adata_sgl.uns['logUMI_para'])
    adata_sgl.uns['corrected'] = 'para'
    
    if achor_group is None or adjust_group is None:
        groups = list(adata_sgl.obs[groupby].unique())
        adjust_group,achor_group = (np.argsort(adata_sgl.uns['logUMI_para'].loc[groups,'mean'])).index
        # adjust the parameter of group with smaller mean by default
        
    achor_mean = adata_sgl.uns['logUMI_para'].loc[achor_group,'mean']
    adjust_mean = adata_sgl.uns['logUMI_para'].loc[adjust_group,'mean']
    
    adata_sgl.uns['shift_ratio'] = 10**(mean_logR - np.abs(achor_mean-adjust_mean))
    
    if achor_mean > adjust_mean:
        mean_corrected = achor_mean - mean_logR
    else:
        mean_corrected = achor_mean + std_logR
        
    adata_sgl.uns['logUMI_para'].loc[adjust_group,'mean'] = mean_corrected

    



#%% preparation in scRNA-only scenario 

def extract_specific_genes(adata_psgl, groupby, pval=0.001, logfc=0):
    '''
    Extract specifically highly expressed genes of each cell type based on DE results.

    Parameters
    ----------
    adata_psgl : AnnData
        UMI matrix of putative singlets (homotypic droplets).
    groupby : 'str'
        The key of the cell type annotation results stored in adata.obs.
    pval : float, optional
        Cutoff of p value to extract significant DE genes. The default is 0.001.
    logfc : float, optional
        Cutoff of log2Fold-change to extract cell-type-specific DE genes. The default is 0.

    Returns
    -------
    degene_sorted : list
        Identified heterotypic doublets composed by the input cell type pair.

    '''
    result = adata_psgl.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    de_df = pd.DataFrame({group + '_' + key[:1]: result[key][group]
                          for group in groups for key in ['names', 'pvals','logfoldchanges']})

    topn = np.min([len([i for i in de_df.index if de_df.loc[i,ct+'_p']<pval and de_df.loc[i,ct+'_l']>logfc]) for ct in groups])
    
    gtmp_list = []
    for n,v in enumerate(groups):
        
        ridx = [gidx for gidx,gname in enumerate(de_df[v+'_n']) if de_df.iloc[gidx,3*n+1]<pval and de_df.iloc[gidx,3*n+2]>0]
        gtmp = de_df.iloc[ridx,n*3:(n+1)*3]
        
        gtmp = gtmp.sort_values(by=[v+'_l'],ascending=False)
        gtmp = gtmp.reset_index(drop=True)
        gtmp_list.append(gtmp.iloc[:topn,:])

    degene_sorted = pd.concat(gtmp_list, 1)
    
    return degene_sorted



from matplotlib import pyplot as plt
import seaborn as sns

def inferHeteroDbl(adata, ct_pair, d_groupby, ct_groupby, g2meta = None, de_sorted=None, mg_gnum=[10,10], threshold = None, vis=True, log=True, return_fig=False):
    '''
    Identification and refinement of heterotypic doublets given specified cell type pair.

    Parameters
    ----------
    adata : AnnData
        UMI matrix of all droplet population.
    ct_pair : list
        Two cell types to investigate their heterotypic doublets.
    d_groupby : str
        The key of the droplet annotation ('Homotypic' or 'Heterotypic') results stored in adata.obs..
    ct_groupby : str
        The key of the cell type annotation results stored in adata.obs..
    g2meta : list, optional
        A list of genes. The default is None.
    de_sorted : data frame, optional
        Output of function 'extract_specific_genes'. The default is None.
    mg_gnum : list, optional
        Number of DEG merged to identify heterodbls. The default is [10,10].
    threshold : TYPE, optional
        Cutoff of expression levels to identify heterodbls. The default is None.
    vis : bool, optional
        Whether to visualize the heterodlb identification results. The default is True.
    log : bool, optional
        Whether to calculate in logarithmic form. The default is True (recommended).
    return_fig : bool, optional
        Whether to return the figure object. The default is False.

    Raises
    ------
    ValueError
        Inspect if genes to be merged for heterotypic doublet detection are specified in a proper way.

    Returns
    -------
    list
        Idnetified heterotypic doublets composed by input cell type pair.

    '''
    if g2meta is not None:
        g2meta_x,g2meta_y = g2meta
    elif de_sorted is not None:
        g2meta_x = de_sorted[ct_pair[0]+'_n'][:mg_gnum[0]]
        g2meta_y = de_sorted[ct_pair[1]+'_n'][:mg_gnum[0]]
    else:
        raise ValueError("value missing. please assign genes to merge by para 'g2meta' or 'de_sorted'")
    
    adata_sgl = adata[adata.obs[d_groupby]=='Homotypic'].copy()
    adata_dbl = adata[adata.obs[d_groupby]=='Heterotypic'].copy()
    
    x_dbl = adata_dbl[:,g2meta_x].X.toarray().sum(1)
    x_sgl = adata_sgl[:,g2meta_x].X.toarray().sum(1)

    y_dbl = adata_dbl[:,g2meta_y].X.toarray().sum(1)
    y_sgl = adata_sgl[:,g2meta_y].X.toarray().sum(1)

    if log:
        x_dbl = np.log(x_dbl+1)
        x_sgl = np.log(x_sgl+1)
        y_sgl = np.log(y_sgl+1)
        y_dbl = np.log(y_dbl+1)

    dbl_sub = pd.DataFrame({'x':x_dbl,'y':y_dbl},index=adata_dbl.obs_names)

    sgl_sub = pd.DataFrame({'x':x_sgl,'y':y_sgl,
                            'celltype_pair':['others']*len(x_sgl)
                           },
                           index=adata_sgl.obs_names)
    sgl_sub.loc[adata_sgl.obs[ct_groupby]==ct_pair[0],'celltype_pair'] = ct_pair[0]
    sgl_sub.loc[adata_sgl.obs[ct_groupby]==ct_pair[1],'celltype_pair'] = ct_pair[1]
    sgl_sub['celltype_pair'] = pd.Categorical(sgl_sub['celltype_pair'].values.tolist(),categories=ct_pair+['others'])
    sgl_sub[ct_groupby] = adata_sgl.obs[ct_groupby]

    if threshold is None:
        ct_x_mgy = sgl_sub.loc[sgl_sub['celltype_pair']==ct_pair[0], 'y']
        ct_x_mgx = sgl_sub.loc[sgl_sub['celltype_pair']==ct_pair[0], 'x']

        ct_y_mgx = sgl_sub.loc[sgl_sub['celltype_pair']==ct_pair[1], 'x']
        ct_y_mgy = sgl_sub.loc[sgl_sub['celltype_pair']==ct_pair[1], 'y']

        mgy_threshold = 1.1*np.max(ct_x_mgy[ct_x_mgx > np.max(ct_y_mgx)])
        mgx_threshold = 1.1*np.max(ct_y_mgx[ct_y_mgy > np.max(ct_x_mgy)])
    else:
        mgx_threshold, mgy_threshold = threshold

    dbl_onhit = [d for d in dbl_sub.index if dbl_sub.loc[d,'x'] > mgx_threshold and dbl_sub.loc[d,'y'] > mgy_threshold]

    if vis:
        
        dbl_sub['pred'] = ['others']*dbl_sub.shape[0]
        dbl_sub.loc[dbl_onhit,'pred'] = '-'.join(ct_pair)
        dbl_sub['pred'] = pd.Categorical(dbl_sub['pred'],categories=['-'.join(ct_pair), 'others'])


        fig = plt.figure(figsize=(11,4),dpi=64)
        sns.set_style('white')
        ax = plt.subplot(1,2,1)
        sns.scatterplot(data=sgl_sub, x='x',y='y',hue='celltype_pair',palette=['blue','green','gray'])
        plt.plot([0,max(sgl_sub['x'])],[mgy_threshold, mgy_threshold],color='blue')
        plt.plot([mgx_threshold, mgx_threshold],[0,max(sgl_sub['y'])],color='green')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15)
        plt.xlabel('MetaDEG of '+ct_pair[0],fontsize=15)
        plt.ylabel('MetaDEG of '+ct_pair[1],fontsize=15)
        plt.title('Identified cell types \nfrom putative singlets',fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax = plt.subplot(1,2,2)
        sns.scatterplot(data=dbl_sub, x='x',y='y',hue='pred',palette=['red','gray'])
        plt.plot([0,max(dbl_sub['x'])],[mgy_threshold, mgy_threshold],color='blue')
        plt.plot([mgx_threshold, mgx_threshold],[0,max(dbl_sub['y'])],color='green')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15)
        plt.xlabel('MetaDEG of '+ct_pair[0],fontsize=15)
        plt.ylabel('MetaDEG of '+ct_pair[1],fontsize=15)
        plt.title('Identified hetero-dbls \nfrom putative doublets',fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        if return_fig:
            return dbl_onhit, fig

    return dbl_onhit




