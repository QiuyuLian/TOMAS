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





def annote_clusters(adata_psgl, ct_mapper, groupby='leiden',anno_group=None):
    '''
    Annotate clusters with domain knowledge of cell  type markers.

    Parameters
    ----------
    adata_psgl : AnnData
        UMI matrix of putative singlets (homotypic droplets).
    ct_mapper : dic
        Domain knowledge of cell type markers. The format must be {'gene1':'celltype1','gene2':'cellltype2','gene3':'celltype3'}.
    groupby : 'str'
        The key of the clustering results stored in adata.obs. The default is 'leiden'.
    anno_group : 'str', optional
        The key of the output annotation of clusters stored in adata.obs. The default is value of 'groupby' with '_anno' as suffix.

    Returns
    -------
    None. The results are saved in adata.obs["groupby"_anno'].

    '''

    if 'rank_genes_groups' not in adata_psgl.uns:
        sc.tl.rank_genes_groups(adata_psgl, groupby, method='wilcoxon')

    result = adata_psgl.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    # ct_mapper = {'g_2426':'A','g_10031':'B','g_5031':'C','g_5497':'D'} 

    # Extract the logFC of the input marker genes in all clusters 
    lf_list = []
    for k in groups:

        gidx = [list(result['names'][k]).index(g) for g in ct_mapper.keys()]
        lf_list.append(result['logfoldchanges'][k][gidx])

    lf_df = pd.DataFrame(np.array(lf_list),index=groups,columns=ct_mapper.keys())
    # row: cluter, col: marker, value: logFC

    clusters = [lf_df.index[np.argmax(lf_df[g])] for g in ct_mapper.keys()]
    annos = dict(zip(clusters, ct_mapper.values()))
    # map clusters to cell types 

    if anno_group is None:
        anno_group = groupby + '_anno'

    adata_psgl.obs[anno_group] = [annos[g] for g in adata_psgl.obs[groupby]]

    print('Cell type annotation is saved in .obs['+anno_group+'].')




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
    
    if 'rank_genes_groups' not in adata_psgl.uns:
        sc.tl.rank_genes_groups(adata_psgl, groupby, method='wilcoxon')
        
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

    



