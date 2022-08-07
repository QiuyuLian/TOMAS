#import copy#, scipy
from tqdm import tqdm
from scipy.stats import norm
#from scipy import sparse
import numpy as np
import scanpy as sc
#import warnings
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator
from scipy.stats import betabinom
from scipy.integrate import quad
from statsmodels.stats import multitest
from scipy import stats
from matplotlib import pyplot as plt
import multiprocessing as mp
import pandas as pd
import time




def total_mRNA_aware_DE(adata_sgl, groupby, reference=None, groups=None, minCells=5,n_cores=None,pval_cutoff=0.05, logFC_cutoff=0):
    '''
    Total-mRNA-aware differential expression analysis.
    Parameters
    ----------
    adata_sgl_rc : AnnData
        The ratio-based corrected UMI count matrix of homo-typic singlets.
        Rows correspond to droplets and columns to genes.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    reference : str
        One droplet category annotated in adata.obs[groupby].
        Compare with respect to the specified group. 
    groups : list, optional
        Droplet categories, e.g. ['Homo-ct1', 'Homo-ct2'] annotated in adata.obs[groupby] to which DE analysis shoudl be performed with. 
        The default is None.
    minCells : int, optional
        Minimum number of cells in one of the group. The default is 5.
    n_cores : int, optional
        Number of cpu cores be used. If not given, generally 80% cpu cores woueld be used.
    pval_cutoff : float, optional
        Significance level. The default is 0.05.
    logFC_cutoff : float, optional
        Cutoff of log2-foldchange to tell the DE significance. The default is 0.
    Returns
    -------
    lrt_DE_dic : dict
        Return total-mRNA-aware DE results in a dictionary. The keys are homo-typic droplet categories compared to the reference.
    '''
    if n_cores is None:
        n_cores = min(30, int(mp.cpu_count()*0.8))

    if groups is None:
        groups = list(adata_sgl.obs[groupby].unique())
    
    if reference is None:
        reference = groups[np.argmin(adata_sgl.uns['logUMI_para'].loc[groups,'mean'])]
    groups.remove(reference)
    groups = [reference]+groups 
    
    #print('Prepare LRT parameters, '+str(time.time()))
    logN_para = adata_sgl.uns['logUMI_para'].values
    alpha_arr = adata_sgl.varm['para_diri'].values.T
    
    alpha_arr[alpha_arr<1e-300] = 1e-300

    ## modify, use para stored in anndata
    bb_para = [np.array([alpha_arr[0], sum(alpha_arr[0])-alpha_arr[0]]),
               np.array([alpha_arr[1], sum(alpha_arr[1])-alpha_arr[1]])]
    
    ## calculate logFC using scanpy without normalizing UMIs
    adata_copy = adata_sgl.copy()
    sc.pp.log1p(adata_copy) 
    sc.tl.rank_genes_groups(adata_copy, groupby, method='wilcoxon', reference=reference)
    DE_logFCref = extract_DE(adata_copy, pval_cutoff=pval_cutoff, logFC_cutoff=logFC_cutoff)

    subdata_list = [adata_sgl[adata_sgl.obs[groupby]==g,:].X.toarray() for g in groups]
    n_cells_each_type = np.column_stack([np.count_nonzero(subdata, axis=0) for subdata in subdata_list])
    n_genes, n_types = n_cells_each_type.shape
    
    goverMcells = [g for g in range(n_genes) if sum(n_cells_each_type[g]>minCells)==n_types]
    
    tot_cells = np.array([subdata.shape[0] for subdata in subdata_list])
    pct_each_type =  n_cells_each_type/tot_cells
    
    #print('run LRT with mp, '+str(time.time()))
    
    ## multiprocessing
    ntasks = len(goverMcells)
    job_unit = ntasks // n_cores
    jobs_sidx = [i*job_unit for i in range(n_cores)]
    jobs_sidx.append(ntasks)
    
    gidx_split = [goverMcells[jobs_sidx[i]:jobs_sidx[i+1]] for i in range(n_cores)]
    
    lrt_DE_dic = {}
    
    for idx in range(1,len(groups)):
    
        group = groups[idx]
    
        print('LRT-based DE: '+group+' vs '+reference+'. This may take a long time. Please wait...')#+', '+str(time.time()))
        data_split = [subdata_list[idx][:,term] for term in gidx_split]
        
        t0 = time.time()
        pool = mp.Pool(n_cores)  
        sim_lr_compact = [pool.apply_async( job_lrtest_sim, (data_split[i],gidx_split[i],bb_para, logN_para, i) ) for i in range(n_cores)]
        pool.close()
        pool.join()
        print('Time cost: '+str(time.time()-t0)+' seconds.')
        
        lr_tmp = [term.get() for term in sim_lr_compact]
        sim_lr_pval = np.array([v for sub in lr_tmp for v in sub])
        _, sim_lrt_pval_adj,_,_ = multitest.multipletests(sim_lr_pval[:,1],method='fdr_bh')
    
        DE_logFCref.index = DE_logFCref[group+'_names']
        DE_logFCref = DE_logFCref.loc[adata_sgl.var_names[goverMcells],:]
        logFC = DE_logFCref[group+'_logfoldchanges']
    
        lrt_df = pd.DataFrame({'log2FC':logFC, 
                               'lrt_stat':sim_lr_pval[:,0],
                               'lrt_pval':sim_lr_pval[:,1],
                               'lrt_pval_adj':sim_lrt_pval_adj,
                               'pct_'+reference: pct_each_type[goverMcells,0],
                               'pct_'+group: pct_each_type[goverMcells,idx],
                              },index=adata_sgl.var_names[goverMcells])
        
        lrt_DE_dic[group] = lrt_df
        
        return lrt_DE_dic

    



def extract_DE(adata,pval_cutoff=0.05,logFC_cutoff=0):
    '''
    Extract DE results from Anndata into a table.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    pval_cutoff : float, optional
        Significance level. The default is 0.05.
    logFC_cutoff : float, optional
        Cutoff of log2-foldchange to tell the DE significance. The default is 0.

    Returns
    -------
    DE_all : pandas.DataFrame
        Organize DE results into a tabel.

    '''
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    DE_all = pd.DataFrame({group + '_' + key: result[key][group] for group in groups for key in ['names', 'pvals','pvals_adj', 'logfoldchanges']})
    
    for group in groups:

        DE_all[group+'_level'] = ['']*DE_all.shape[0]
        up = [g for g in DE_all.index if DE_all.loc[g,group+'_pvals_adj'] < pval_cutoff and DE_all.loc[g,group+'_logfoldchanges'] > logFC_cutoff]
        dn = [g for g in DE_all.index if DE_all.loc[g,group+'_pvals_adj'] < pval_cutoff and DE_all.loc[g,group+'_logfoldchanges'] < -logFC_cutoff]
        ns = [g for g in DE_all.index if g not in up+dn]#DE_all.loc[g,group+'_pvals_adj'] > pval_cutoff]

        DE_all.loc[up, group+'_level'] = 'up'
        DE_all.loc[dn, group+'_level'] = 'dn'
        DE_all.loc[ns, group+'_level'] = 'ns'
    
    return DE_all



def job_lrtest_sim(data, gidx, bb_para, logN_para, i):
    #print(gidx[0])
    #print('core '+str(i)+', start at '+str(time.time()))
    lr_list = []
    for g in tqdm(range(data.shape[1])):

        k_arr = data[:,g].astype(int)
        
        kpmf0 = k_marg_pmf(k_arr, bb_para, logN_para, c=0, g=gidx[g])
        ll_k0 = ll_kmarg(k_arr, kpmf0)
       
        kpmf1 = k_marg_pmf(k_arr, bb_para, logN_para, c=1, g=gidx[g])
        ll_k1 = ll_kmarg(k_arr, kpmf1)

        lrt = -2*(ll_k0-ll_k1)
        pval = stats.chi2.sf(lrt, 4)
        lr_list.append([lrt,pval])

    return lr_list



def ll_kmarg(k_arr, k_pmf,g=None):
    
    '''
    with warnings.catch_warnings(record=True) as caught_warnings:
        k_pmf[k_pmf==0]=1e-300
        val = np.sum([np.log(k_pmf[k]) for k in k_arr])
    for warn in caught_warnings:
        print(g)
    '''
    k_pmf[k_pmf==0]=1e-300
    val = np.sum([np.log(k_pmf[k]) for k in k_arr])       

    return val


def integrand(logn,k, alpha,beta,mu,sigma):
    ll_tmp = betabinom.logpmf(k,10**logn,alpha,beta) + norm.logpdf(logn,mu,sigma)
    return np.exp(ll_tmp)

def k_marg_pmf(k_arr, bb_para, logN_para, c, g):
    alpha, beta = bb_para[c][:,g]
    mu,sigma = logN_para[c]
    k_max = np.max([int(10**(mu+3*sigma) * alpha/(alpha+beta)), int(3*np.max(k_arr))])
    k_pmf_r = [quad(integrand, mu-3*sigma,mu+3*sigma,args=(k,alpha,beta,mu,sigma))[0] for k in range(k_max)]
    return np.array(k_pmf_r)/sum(k_pmf_r)


def lrt_onegene(adata, bb_para, logN_para, g, display=False):
    
    k_arr = adata.X[:,g].astype(int)
    #print(type(k_arr))
    kpmf0 = k_marg_pmf(k_arr, bb_para, logN_para, c=0, g=g)
    ll_k0 = ll_kmarg(k_arr, kpmf0)

    kpmf1 = k_marg_pmf(k_arr, bb_para, logN_para, c=1, g=g)
    ll_k1 = ll_kmarg(k_arr, kpmf1)

    lrt = -2*(ll_k0-ll_k1)
    pval = stats.chi2.sf(lrt, 4)

    if display:
        
        values, counts = np.unique(k_arr, return_counts=True)
        
        fig, axs =plt.subplots(1,3,figsize=(10,4),dpi=64)
        plt.subplot(1,2, 1)

        plt.vlines(values, 0, counts/sum(counts), color='gray',alpha=0.5, lw=4)
        plt.scatter(range(len(kpmf0)),kpmf0)
        plt.xlabel('k',fontsize=15)
        plt.ylabel('Density',fontsize=15)
        plt.title('H0, ll='+str(round(ll_k1,2)),fontsize=18)

        plt.subplot(1,2, 2)
        #values, counts = np.unique(k_arr2, return_counts=True)
        plt.vlines(values, 0, counts/sum(counts), color='gray',alpha=0.5, lw=4)
        plt.scatter(range(len(kpmf1)),kpmf1)
        plt.xlabel('k',fontsize=15)
        plt.ylabel('Density',fontsize=15)
        plt.title('H1, ll='+str(round(ll_k1,2)),fontsize=18)

        plt.suptitle(adata.var_names[g]+', lrt='+str(round(lrt))+', pval='+str(pval),fontsize=20)
        plt.tight_layout()
        plt.show()

    return [lrt,pval]



def summarize2DE(gs_df, lrt_df, group, pval_cutoff = 0.05, logFC_cutoff = 0):
    '''
    Compare and summarize two DE results.

    Parameters
    ----------
    gs_df : pandas.DataFrame
        Global-scaling DE table.
    lrt_df : pandas.DataFrame
        LRT-based DE table of TOMAS.
    group : str
        The category of droplet to compare with reference.
    pval_cutoff : float, optional
        Significance level. The default is 0.05.
    logFC_cutoff : float, optional
        Cutoff of log2-foldchange to tell the DE significance. The default is 0.

    Returns
    -------
    de_df : pandas.DataFrame
        Table fusing two input DE resutls.

    '''
    
    gs_df.index = gs_df[group+'_names']
    gs_df = gs_df.loc[lrt_df.index,:]
    
    de_df = pd.DataFrame({'log2FC_gs':gs_df[group+'_logfoldchanges'],
                          'pval_adj_gs':gs_df[group+'_pvals_adj'],
                          'level_gs':['']*lrt_df.shape[0],
                          'log2FC_rc':lrt_df['log2FC'],
                          'pval_adj_rc':lrt_df['lrt_pval_adj'],
                          'level_rc':['']*lrt_df.shape[0]
                         },index=lrt_df.index)

    up = [g for g in gs_df.index if gs_df.loc[g, group+'_pvals_adj'] < pval_cutoff and gs_df.loc[g,group+'_logfoldchanges'] > logFC_cutoff]
    dn = [g for g in gs_df.index if gs_df.loc[g, group+'_pvals_adj'] < pval_cutoff and gs_df.loc[g,group+'_logfoldchanges'] < logFC_cutoff]
    ns = [g for g in gs_df.index if gs_df.loc[g, group+'_pvals_adj'] > pval_cutoff]

    de_df.loc[up,'level_gs'] = 'up'
    de_df.loc[dn,'level_gs'] = 'dn'
    de_df.loc[ns,'level_gs'] = 'ns'


    up = [g for g in lrt_df.index if lrt_df.loc[g,'lrt_pval_adj'] < pval_cutoff and lrt_df.loc[g,'log2FC'] > logFC_cutoff]
    dn = [g for g in lrt_df.index if lrt_df.loc[g,'lrt_pval_adj'] < pval_cutoff and lrt_df.loc[g,'log2FC'] < logFC_cutoff]
    ns = [g for g in lrt_df.index if lrt_df.loc[g,'lrt_pval_adj'] > pval_cutoff]

    de_df.loc[up,'level_rc'] = 'up'
    de_df.loc[dn,'level_rc'] = 'dn'
    de_df.loc[ns,'level_rc'] = 'ns'

    lev1 = de_df['level_gs'].values
    lev2 = de_df['level_rc'].values

    lel_compare = [lev1[i]+'2'+lev2[i] for i in range(len(lev1))]
    vals, cnts = np.unique(lel_compare,return_counts=True)

    DE_level_delta = pd.DataFrame(np.zeros([3,3]),index=['up','ns','dn'],columns=['up','ns','dn'])

    for i in range(len(vals)):
        ori,rc = vals[i].split('2')
        DE_level_delta.loc[ori,rc] = cnts[i]
    
    DE_level_delta = DE_level_delta.astype(int)
    print(DE_level_delta)

    de_df['levelchange'] = [de_df['level_gs'][i]+'2'+de_df['level_rc'][i] for i in range(de_df.shape[0])]

    return de_df,DE_level_delta
