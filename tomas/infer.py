#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 03:47:01 2022

@author: qy
"""
import time, os
import numpy as np
from scipy.stats import gaussian_kde, norm, dirichlet,multinomial
import multiprocessing as mp
import pandas as pd
import scipy
import scanpy as sc
import copy
from sklearn.mixture import GaussianMixture

#from auxi import rm_outliers
def rm_outliers(x):
    iqr = scipy.stats.iqr(x)
    outlier_lb = np.quantile(x,0.25)-1.5*iqr
    outlier_ub = np.quantile(x,0.75)+1.5*iqr
    x_shrinkage = x[x > outlier_lb]
    x_shrinkage = x_shrinkage[x_shrinkage<outlier_ub]
    return x_shrinkage#,(outlier_lb,outlier_ub)

#%% Infer Total-mRNA Ratios 


def ratios_bc(adata_mg, dblgroup, max_iter=100, tol=1e-3, n_logR=100, n_p=1000, warm_start=False, verbose=0, verbose_interval=2):
    '''
    Infer the total-mRNA ratio between two cell types given the heterotypic doublets composed by the two cell types.

    Parameters
    ----------
    adata_mg : AnnData
       The UMI count matrix of hetero-doublets in metagenes.
       Rows correspond to droplets and columns to metagenes.
    dblgroup : str
        Name of the heterotypic doublets in the form of 'A_B'.
    max_iter : int, optional
        The maximal number of iterations to perform. The default is 20.
    tol : float, optional
        The convergence threshold. The iterations will stop when the lower bound log-likelihood gain is below this threshold. The default is 1e-3.
    n_logR : int, optional
        The number of R to be sampled per time. The default is 100.
    n_p : int, optional
        The number of p to be sampled per time. The default is 1000.
    warm_start : bool, optional
        If "warm_start" is True, the parameters of the last running is used as initialization for the current call. The default is False.
    verbose : bool, optional
        Enable verbose output. The default is 0, which means no output printed. If 1 then it prints the log-likelihood gain and time lapse for iteration step.
    verbose_interval : int, optional
        Number of iteration done before the next print. The default is 2.

    Returns
    -------
    None. The results are saved in .uns['ratio'] of the input AnnData.

    '''
    
    alpha0,alpha1 = adata_mg.varm['para_diri'].transpose()
    Y = adata_mg.X
    n = len(Y)
    num_trials = Y.sum(1).astype(int)
    
    # Initialization 
    if warm_start:
        
        if 'ratio' not in adata_mg.uns:
            raise ValueError('No fitting history. Run again with "warm_start=False".')

        ratio_dic = adata_mg.uns['ratio']
        log2R_m, log2R_std = ratio_dic['log2R_para']
        
        p0_best = ratio_dic['p_'+dblgroup.split('_')[0]]
        p1_best = ratio_dic['p_'+dblgroup.split('_')[1]]
        w_best = ratio_dic['w_best']
        ll_best = ratio_dic['ll_best']
        log2R_std_prior = ratio_dic['log2R_std_prior']
 
    else:
    
        w_prior = initialize_w(Y, alpha0, alpha1)
        log2R_m = np.log2((1-w_prior)/w_prior)
        #log2R_std_prior = adata_mg.uns['logUMI_para'].loc[dblgroup.split('_'),'std'].sum()
        #log2R_std = copy.deepcopy(log2R_std_prior)
        log2R_std = adata_mg.uns['logUMI_para'].loc[dblgroup.split('_'),'std'].sum()
        log2R_std_prior = np.max([1,log2R_std])

        p0_best = np.array([alpha0/alpha0.sum()]*n)
        p1_best = np.array([alpha1/alpha1.sum()]*n)
        #w_best = np.array([w_prior]*n)
        logR_sampling = np.random.normal(log2R_m, log2R_std, n)
        w_best = 1/(2**logR_sampling+1)
        
        #p_dbl_best = alpha0/alpha0.sum()*w_prior+alpha1/alpha1.sum()*(1-w_prior)
        p_dbl_best = np.array([alpha0/alpha0.sum()*w + alpha1/alpha1.sum()*(1-w) for w in w_best])
        
        ll_best = np.array([[dirichlet.logpdf(alpha0/alpha0.sum(),alpha0)]*n, 
                            [dirichlet.logpdf(alpha1/alpha1.sum(),alpha1)]*n,
                            #[norm.logpdf(log2R_m,log2R_m,log2R_std)]*n,
                            list(norm.logpdf(logR_sampling,log2R_m,log2R_std)),
                            [multinomial.logpmf(Y[j],num_trials[j],p_dbl_best[j]) for j in range(n)]])
        
        ratio_dic = {}
        ratio_dic['p_'+dblgroup.split('_')[0]] = p0_best
        ratio_dic['p_'+dblgroup.split('_')[1]] = p1_best
        ratio_dic['w_best'] = w_best
        ratio_dic['log2R_para'] = [log2R_m, log2R_std]
        ratio_dic['R_est'] = 2**log2R_m
        ratio_dic['ll_best'] = ll_best.sum(0)
        ratio_dic['niter'] = 0
        adata_mg.uns['ratio'] = ratio_dic
            
            
    #log2R_std_prior = np.max([1,log2R_std_prior])
    ll_tot_best = ll_best.sum()
    delta_ll = np.inf
    print('Initialized.')
    
    #niter = 0
    niter_NoUpdate = 0
    t0 = time.time()
    ll = copy.deepcopy(ll_tot_best)
    for niter in range(max_iter):

        NoUpdate = 0
        ##### update w, fix p
        logR_sampling = np.random.normal(log2R_m, log2R_std_prior, n_logR)
        logR_ll = norm.logpdf(logR_sampling, log2R_m, log2R_std_prior) 
        W = 1/(2**logR_sampling+1)

        p_dbl_fixp = np.array([p0_best*w + p1_best*(1-w) for w in W])

        mult_ll_allsampling = np.array([[multinomial.logpmf(Y[j],num_trials[j],p_dbl_fixp[i,j]) for i in range(n_logR)] for j in range(n)])
        ll_mult_logR = mult_ll_allsampling+logR_ll
        ll_mult_logR_max = ll_mult_logR.max(1)

        midx = ll_mult_logR.argmax(1)
        w_best_tmp = W[midx] 
        mult_ll_max = np.array([mult_ll_allsampling[i][midx[i]] for i in range(n)])
        logR_ll_max = logR_ll[midx]

        replace = ll_mult_logR_max > ll_best[2:].sum(0)
        if sum(replace) > 0:

            w_best[replace] = w_best_tmp[replace]
            log2R_m,log2R_std = norm.fit(rm_outliers(np.log2((1-w_best)/w_best)))
            #log2R_m,log2R_std = norm.fit(np.log2((1-w_best)/w_best))
            if log2R_std < 0.01: #
                log2R_std = np.max([log2R_std,log2R_std_prior])

            ll_best[2,replace] = logR_ll_max[replace]
            ll_best[3,replace] = mult_ll_max[replace]

            ll_tot = ll_best.sum()
            delta_ll = ll_tot - ll_tot_best
            ll_tot_best = ll_tot
        else:    
            NoUpdate += 0.5

        ##### update p, fix w
        p0 = np.random.dirichlet(alpha0, n_p)
        diri0_ll = np.array([dirichlet.logpdf(p, alpha0) for p in p0])

        p1 = np.random.dirichlet(alpha1, n_p)
        diri1_ll = np.array([dirichlet.logpdf(p, alpha1) for p in p1])

        p_dbl_fixw = np.array([p0*w + p1*(1-w) for w in w_best])

        mult_ll_allsampling = np.array([[multinomial.logpmf(Y[j],num_trials[j],p_dbl_fixw[j,i]) for i in range(n_p)] for j in range(n)])
        ll_diri_mult = mult_ll_allsampling+diri0_ll+diri1_ll
        ll_diri_mult_max = ll_diri_mult.max(1)

        midx = ll_diri_mult.argmax(1)
        p0_best_tmp = p0[midx]
        p1_best_tmp = p1[midx]
        mult_ll_max = np.array([mult_ll_allsampling[i,midx[i]] for i in range(n)])
        diri0_ll_max = diri0_ll[midx]
        diri1_ll_max = diri1_ll[midx]

        replace = ll_diri_mult_max > ll_best[[0,1,3]].sum(0)

        if sum(replace) > 0:

            p0_best[replace] = p0_best_tmp[replace]
            p1_best[replace] = p1_best_tmp[replace]

            ll_best[0,replace] = diri0_ll_max[replace]
            ll_best[1,replace] = diri1_ll_max[replace]
            ll_best[3,replace] = mult_ll_max[replace]

            ll_tot = ll_best.sum()
            delta_ll = ll_tot - ll_tot_best
        else:
            NoUpdate += 0.5

        if NoUpdate == 1:
            niter_NoUpdate += 1
        else:
            niter_NoUpdate = 0
         
        if verbose > 0 and (niter+1) % verbose_interval == 0:
            print('Iteration '+str(niter+1)+'\ttime lapse '+str(time.time()-t0)+'s\tll change '+str(ll_best.sum()-ll)+'\n')
            t0 = time.time()
            ll = ll_best.sum()
        if niter_NoUpdate >= 3 or delta_ll < tol:
            break
    
    # update the info stored in anndata
    adata_mg.uns['ratio']['p_'+dblgroup.split('_')[0]] = p0_best
    adata_mg.uns['ratio']['p_'+dblgroup.split('_')[1]] = p1_best
    adata_mg.uns['ratio']['w_best'] = w_best
    adata_mg.uns['ratio']['log2R_para'] = [log2R_m, log2R_std]
    adata_mg.uns['ratio']['R_est'] = 2**log2R_m
    adata_mg.uns['ratio']['ll_best'] = ll_best
    adata_mg.uns['ratio']['niter'] += niter

    ## check confidence
    r_list = (1-w_best)/w_best
    if len(r_list) < 15:
        log2r = np.log2(r_list)
    else:
        log2r = rm_outliers(np.log2(r_list))

    unimodal = GaussianMixture(1).fit(log2r.reshape(-1,1))
    bimodal =  GaussianMixture(2).fit(log2r.reshape(-1,1))
    lrt = -2*(unimodal.score_samples(log2r.reshape(-1,1)).sum() - bimodal.score_samples(log2r.reshape(-1,1)).sum())
    pval = scipy.stats.chi2.sf(lrt, 3)
    adata_mg.uns['ratio']['confidence'] = pval



    
from scipy import optimize

def func(x, a):
    y = a*x 
    return y


def initialize_w(Y, alpha1, alpha2):
    
    p1 = alpha1/sum(alpha1)
    p2 = alpha2/sum(alpha2)
    gsum_mix = Y.sum(0)
    pmix = gsum_mix/sum(gsum_mix)
    # method=='ols'
    n = Y.shape[0]
    w_ = optimize.curve_fit(func, 
                            xdata = np.ravel(np.array([p1]*n)-np.array([p2]*n)), 
                            ydata = np.ravel(pmix - np.array([p2]*n)))

    w = w_[0][0]
    
    if w >= 1 or w <= 0:
    # method=='kde'
        idx_in = [i for i in range(len(p1)) if pmix[i] > min(p1[i],p2[i]) and pmix[i] < max(p1[i], p2[i])]
        
        w_tmp = [(pmix[i]-p2[i])/(p1[i]-p2[i]) for i in idx_in if p1[i] - p2[i] != 0]
        
        kde = gaussian_kde(w_tmp)
        samples = np.linspace(0, 1, 1000)
        probs = kde.evaluate(samples)
        maxima_index = probs.argmax()
        w = samples[maxima_index]

    return w




#%% Extract Exclusive Meta-Genes  

def get_dbl_mg(adata,groupby,groups='all',output=None, num_mg = 100, kl_cutoff=None, merging_threshold=5, skip_threshold=2,alphaMin = 1):
    '''
    Extract metagenes from raw UMI counts of heterotypic doublets.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    groups : list or str ('all'), optional
        Groups of heterodbls to extract metagenes. List of a groups of valid dbl names or a string 'all'. The default is 'all'.
    output : path
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

    Raises
    ------
    ValueError
        Check the legality of argements.

    Returns
    -------
    adata_mgdic : dic
        Keys are names of heterotypic doublets. Values are UMI counts in metagenes of heterotypic doublets.

    '''
    if isinstance(groups, str):
        if groups == 'all':
            #groups = adata.obs[groupby].unique().tolist()
            groups = [d for d in adata.obs[groupby].unique() if len(d.split('_'))==2]
            if 'unknown' in groups:
                groups.remove('unknown')
        else:
            raise ValueError('Invalid input. If "groups" is a string, it can only be "all". If you want to specify a single heterodbl group, put it in a list, e.g. ["B_CD8T"].')
    elif isinstance(groups, list):
        for f in groups:
            if f not in adata.obs[groupby].unique():
                raise ValueError('Invalid input. The annotation of heterodbls, set by argument "groupby" in adata.obs, do not contain "'+f+'".')
    else:
        raise ValueError('Invalid input of argument "groupby".')
        
    if isinstance(kl_cutoff,float):
        if 'kl' not in adata.varm.keys():
            raise ValueError('KL divergences have not be calculated yet. Run "get_dbl_mg" with "kl_cutoff" set as "None" to disabe filtering genes by KL (recommended) or first calculate the KL divergens')

    adata_mgdic = {}
    for dblgroup in groups:
        mg_tmp = get_dbl_mg_bc(adata, 
                               groupby, 
                               dblgroup, 
                               output=output,
                               num_mg=num_mg, 
                               kl_cutoff=kl_cutoff,
                               merging_threshold=merging_threshold,
                               skip_threshold=skip_threshold,
                               alphaMin=alphaMin)
        adata_mgdic[dblgroup] = mg_tmp

    return adata_mgdic
    





def get_dbl_mg_bc(adata, groupby, dblgroup, output=None, num_mg = 100, kl_cutoff=None, merging_threshold=5, skip_threshold=2,alphaMin = 1):
    '''
    Extract metagenes from raw UMI counts of a certain type of heterotypic doublets.
    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    groups : list of strings
        Specify two droplet categories, e.g. ['Homo-ct1', 'Homo-ct2'] annotated in adata.obs[groupby], which form the hetero-dbl of interest.
    output : path
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
    adata_dbl_mg_top : AnnData
        The UMI count matrix of shape `n_obs` × `n_metangenes`.
        Rows correspond to droplets and columns to metagenes.
    '''
    groups = dblgroup.split('_') #+[dblgroup]

    adata_dbl = adata[adata.obs[groupby]==dblgroup,:].copy()
    adata_dbl = adata_dbl[:,adata_dbl.X.sum(0)>0]


    #count_dbl = adata_dbl[adata_dbl.obs[groupby]==dblgroup,:].X
    count_dbl = adata_dbl.X
    if isinstance(count_dbl, scipy.sparse.csr_matrix):
        count_dbl = count_dbl.toarray()


    raw_Alpha1_0 = adata_dbl.varm['para_diri'][groups[0]]
    raw_Alpha2_0 = adata_dbl.varm['para_diri'][groups[1]]


    if isinstance(kl_cutoff,float):
        # use genes with KL greater than cutoff to obtain meta-genes
        kl = adata_dbl.varm['kl']
        KL_filter_idx = np.where(kl > kl_cutoff)[0]
    else:
        KL_filter_idx = np.array([i for i in range(len(raw_Alpha1_0))])

    raw_Alpha1 = raw_Alpha1_0[KL_filter_idx]
    raw_Alpha2 = raw_Alpha2_0[KL_filter_idx]

    mg_dic = get_mg(raw_Alpha1, raw_Alpha2, merging_threshold, skip_threshold,alphaMin)

    mg_pool = mg_dic['mg_pool'] 

    mg_gidx = [list(KL_filter_idx[term]) for term in mg_pool]
    mg_genepool = [g for sub in  mg_gidx  for g in sub]

    mg_alpha1 = mg_dic['alpha1']
    mg_alpha2 = mg_dic['alpha2']

    left_gidx = list(set(range(len(raw_Alpha1_0))).difference(mg_genepool)) 
    mg_gidx.append(left_gidx)

    # merge alpha and counts according to meta-genes

    mg_Alpha_arr = np.concatenate((np.array([mg_alpha1, mg_alpha2]), np.array([[sum(raw_Alpha1_0)-sum(mg_alpha1)],[sum(raw_Alpha2_0)-sum(mg_alpha2)]])),axis=1)
    Y = np.transpose(np.array([np.sum(count_dbl[:,term],1) for term in mg_gidx]))

    adata_dbl_mg = sc.AnnData(Y,dtype=Y.dtype)
    adata_dbl_mg.obs_names = adata_dbl.obs_names#[adata_dbl.obs[groupby]==dblgroup]
    adata_dbl_mg.varm['para_diri'] = mg_Alpha_arr.transpose()
    if output is not None:
        if not os.path.exists(output):
            os.makedirs(output)
        adata_dbl_mg.write_h5ad(os.path.join(output,dblgroup+'_dbl_mg.h5ad'))

    # extract top num_mg 
    topmg_idx = [[i] for i in range(num_mg)] + [list(range(num_mg,  len(mg_gidx)))]
    mg_Alpha_new = np.transpose(np.array([np.sum(mg_Alpha_arr[:,term],1) for term in topmg_idx]))
    Y_new = np.transpose(np.array([np.sum(Y[:,term],1) for term in topmg_idx]))

    adata_dbl_mg_top = sc.AnnData(Y_new, dtype=Y_new.dtype)
    adata_dbl_mg_top.obs_names = adata_dbl.obs_names#[adata.obs[groupby]==dblgroup]
    adata_dbl_mg_top.varm['para_diri'] = mg_Alpha_new.transpose()
    if len(adata.uns.keys()):
        for val in adata.uns.keys():
            adata_dbl_mg_top.uns[val] = adata.uns[val]

    return adata_dbl_mg_top





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





#%% preparation in scRNA-only scenario 


from matplotlib import pyplot as plt
import seaborn as sns
import itertools

def heteroDbl(adata, d_groupby, ct_groupby, de_sorted=None,dbl_groupby = 'heteroDbl',vis=False):
    '''
    Identification and refinement of heterotypic doublets

    Parameters
    ----------
    adata : AnnData
        UMI matrix of all droplet population.
    d_groupby : str
        The key of the droplet annotation ('Homotypic' or 'Heterotypic') results stored in adata.obs.
    ct_groupby : str
        The key of the cell type annotation results stored in adata.obs.
    de_sorted : data frame, optional
        Output of function 'extract_specific_genes'. The default is None.
    dbl_groupby : TYPE, optional
        DESCRIPTION. The default is 'heteroDbl'.
    vis : bool, optional
        Whether to visualize the heterodlb identification results. The default is True.

    Returns
    -------
    None. Identified heterotypic 

    '''
    
    groups = adata.obs[ct_groupby].unique().tolist()

    if 'unknown' in groups:
        groups.remove('unknown')

    #if dbl_groupby not in adata.obs:
    adata.obs[dbl_groupby] = 'unknown'
        
    for pair in itertools.combinations(groups,2):
        dbl = '_'.join(pair)
        print('Detect doublets composed by cell type '+pair[0]+' and '+pair[1]+'.')
        dbl_candidate = heteroDbl_bc(adata, dbl,  d_groupby = d_groupby, ct_groupby = ct_groupby, de_sorted = de_sorted,vis=vis)
        adata.obs.loc[dbl_candidate,dbl_groupby] = dbl  
        print(str(len(dbl_candidate))+' putative heterotypic doublets detected.')        



def heteroDbl_bc(adata, dbl, d_groupby, ct_groupby, g2meta = None, de_sorted=None, mg_gnum=[10,10], threshold = None, threshold_x=None,threshold_y=None,vis=True, log=True, return_fig=False):
    '''
    Identification and refinement of a certain type of heterotypic doublets specified by "dbl".

    Parameters
    ----------
    adata : AnnData
        UMI matrix of all droplet population.
    dbl : str
        Heterotypic dboulet group to be investigated, named by two cell types linked by '_'.
    d_groupby : str
        The key of the droplet annotation ('Homotypic' or 'Heterotypic') results stored in adata.obs.
    ct_groupby : str
        The key of the cell type annotation results stored in adata.obs.
    g2meta : list, optional
        A list of genes. The default is None.
    de_sorted : data frame, optional
        Output of function 'extract_specific_genes'. The default is None.
    mg_gnum : list, optional
        Number of DEG merged to identify heterodbls. The default is [10,10].
    threshold : list, optional
        Cutoff of expression levels in current cell type pair to identify heterodbls. The default is None.
    threshold_x : float, optional
        Cutoff of expression level in current (x axis) cell type to identify heterodbls. The default is None.
    threshold_y : float, optional
        Cutoff of expression level in current (y axis) cell type to identify heterodbls. The default is None.
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
    ct_pair = dbl.split('_')
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

        mgy_threshold = np.max(ct_x_mgy[ct_x_mgx > np.max(ct_y_mgx)])
        mgx_threshold = np.max(ct_y_mgx[ct_y_mgy > np.max(ct_x_mgy)])
        
        mgx_threshold += np.max([0.1*mgx_threshold, 0.2]) # add a margin
        mgy_threshold += np.max([0.1*mgy_threshold, 0.2]) # add a margin
        
    else:
        mgx_threshold, mgy_threshold = threshold
        
    if threshold_x is not None:
        mgx_threshold = threshold_x
        
    if threshold_y is not None:
        mgy_threshold = threshold_y
    

    dbl_onhit = [d for d in dbl_sub.index if dbl_sub.loc[d,'x'] > mgx_threshold and dbl_sub.loc[d,'y'] > mgy_threshold]

    if vis:
        
        dbl_sub['pred'] = ['others']*dbl_sub.shape[0]
        dbl_sub.loc[dbl_onhit,'pred'] = '_'.join(ct_pair)
        dbl_sub['pred'] = pd.Categorical(dbl_sub['pred'],categories=['_'.join(ct_pair), 'others'])


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






#%% old version of inferring ratios

def ratio_2types(adata,groupby,groups,output,subset=100,nrepeat=10):
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

    Returns
    -------
    esti_r_list : numpy.ndarray
        Estimate of total mRNA ratio of each hetero-dbouelt.

    '''
    if not os.path.exists(output):
        os.makedirs(output)
        
    name=groups[2]
    logpath = os.path.join(output,'_'.join(['rlog',name]))
    if not os.path.exists(logpath):
        os.makedirs(logpath)
        
    
    adata_dbl_mg = get_dbl_mg_bc(adata,groupby,groups)
    
    mg_Alpha_new = adata_dbl_mg.varm['para_diri'].transpose()
    Y_new = adata_dbl_mg.X
    
    if subset is not None:
        if Y_new.shape[0] > subset:
            idx = np.random.choice(Y_new.shape[0], subset, replace=False)
            Y_new = Y_new[idx,:]
    
    '''
    ### use raw data to init
    count_dbl = adata[adata.obs[groupby]==groups[2],:].X
    if isinstance(count_dbl, scipy.sparse.csr.csr_matrix):
        count_dbl = count_dbl.toarray()
         
    w_init = initialize_w(count_dbl, adata.varm['para_diri'][groups[0]].values, adata.varm['para_diri'][groups[1]].values)
    ###
    '''
    #w_init = initialize_w(Y_new, mg_Alpha_new[0], mg_Alpha_new[1])
    #estimateW(mg_Alpha_new,Y_new,logpath,w_init,nrepeat,name)
    estimateW(mg_Alpha_new,Y_new,logpath,nrepeat=nrepeat,name=name)

    
    ll_list = []
    for ridx in range(nrepeat):
        RandLL = pd.read_csv(os.path.join(logpath,'.'.join(['RandLL',name,str(ridx),'csv'])),header=0,index_col=0)
        ll_list.append(RandLL['LL'].values[-1])
    
    ll_argmax = np.argmax(ll_list)
    #print(ll_argmax)

    ridx = ll_argmax
    
    R_est0 = pd.read_csv(os.path.join(logpath,'.'.join(['Rtrack',name,str(ridx),'csv'])),header=0,index_col=0)
    esti_r_list = R_est0.iloc[:,-1].values
    
    #w_corrected = [ratio_correction(1/(r+1),left_mg_pmass[ct][0],left_mg_pmass[ct][1]) for r in esti_r_list]
    #r_list = [(1-w)/w for w in esti_r_list]
    
    return esti_r_list



def estimateW(mg_Alpha_new,Y_new,output,w_init=None,nrepeat=10,name='',n_cores=None):

    ndoublets = len(Y_new)

    Alpha_mat = mg_Alpha_new
    
    alpha0 = Alpha_mat[0]
    alpha1 = Alpha_mat[1]
    
    if w_init is None:
        w_init = initialize_w(Y_new, alpha0, alpha1)
        
    for repeat in range(nrepeat):
    
        log = open(os.path.join(output,'.'.join(['log',name,str(repeat),'txt'])),'w',buffering=1)
    
        t0 = time.time() 
        log.write('start time '+str(t0)+'\n')
    
        # initialize  
        r_init = 2**np.random.normal(np.log2((1-w_init)/w_init),1)
        log.write('initizalized R '+str(r_init)+'\n')
        logR_m_record = []
        LL_record = []
        r_track = []
    
        logR_m = np.log2(r_init)
        logR_std = 1
        logR_m_record.append(logR_m)
    
        num_p = 1000
        num_w = 100
        maxiter = 2
        delta_LL_tol = 0.1
        job_unit = 20
        if n_cores is None:
            n_cores = min(50, int(mp.cpu_count()*0.8))
        
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
    
            tp0 = time.time()
            if parallel:
    
                pool = mp.Pool(n_cores)  
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
        R_LL_df.to_csv(os.path.join(output,'.'.join(['RandLL',name,str(repeat),'csv'])))
    
        ## record r estimated for each doublet per update
        r_track_df = pd.DataFrame(np.array(r_track).transpose(),columns=range(len(r_track)))
        r_track_df.to_csv(os.path.join(output,'.'.join(['Rtrack',name,str(repeat),'csv'])))
     


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
        
        d_p0_sampling  = np.random.dirichlet(alpha0, num_p)
        d_p0_sampling = get_P_valid(d_p0_sampling)
        diri_ll_0 = [dirichlet.logpdf(p, alpha0)  - fp_dll0 for p in d_p0_sampling]
        
        d_p1_sampling  = np.random.dirichlet(alpha1, num_p)
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
    
    P[P==0] = 1e-10 

    return P
