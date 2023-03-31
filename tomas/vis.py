#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 02:29:17 2022

@author: qy
"""

import numpy as np
import seaborn as sns    
import scipy 
#import matplotlib
import copy
import pandas as pd
import os
import itertools
from matplotlib import pyplot as plt
import warnings

#from auxi import rm_outliers
def rm_outliers(x):
    iqr = scipy.stats.iqr(x)
    outlier_lb = np.quantile(x,0.25)-1.5*iqr
    outlier_ub = np.quantile(x,0.75)+1.5*iqr
    x_shrinkage = x[x > outlier_lb]
    x_shrinkage = x_shrinkage[x_shrinkage<outlier_ub]
    return x_shrinkage#,(outlier_lb,outlier_ub)


def dmn_convergence(group,output,return_fig=None):
    '''
    Visualize the convergence of DMN optimation process.

    Parameters
    ----------
    group : str
        The droplet category to display the optimization process.
    output : path
        The specified output path when calling fucntion 'tomas.fit.dmn'.

    Returns
    -------
    None.

    '''
    log = pd.read_csv(os.path.join(output,group+'.dmnlog.txt'))

    fig, axs =plt.subplots(1,3,figsize=(12,3),dpi=64)

    plt.subplot(1,3,1)
    plt.scatter(log.index, log['loglik'], s=1)
    plt.xlabel('iteration')
    plt.ylabel('logLikelihood')
    plt.title('logLikelihood',fontsize=14)

    plt.subplot(1,3,2)
    plt.scatter(log.index, log['alpha_sum'], s=1)
    plt.xlabel('iteration')
    plt.ylabel('l1-norm of alpha')
    plt.title('precision',fontsize=14)

    plt.subplot(1,3,3)
    plt.scatter(log.index, log['alpha_l2norm_delta'], s=1)
    plt.xlabel('iteration')
    plt.ylabel('delta ||alpha||2')
    plt.title('delta of l2norm of alpha',fontsize=14)

    fig.tight_layout()
    if return_fig is True:
        return fig
    plt.show()




'''
def alpha_opt_process(record, saveFig = False, filename=None):

    
    if filename is None and saveFig:
        raise ValueError("Provide a filename in the form of '/path_to_save_the_file/name_of_the_file' without suffix!")

    fig, axs =plt.subplots(2,2,figsize=(9,6),dpi=128)
    
    plt.subplot(2,2,1)
    term = record['LL']
    plt.scatter(range(len(term)-2), term[2:], s=1)
    plt.xlabel('iteration')
    plt.ylabel('logLikelihood')
    plt.title('logLikelihood',fontsize=14)
    
    plt.subplot(2,2,2)
    term = record['alpha_norm']
    plt.scatter(range(len(term)), term, s=1)
    plt.xlabel('iteration')
    plt.ylabel('alpha l2-norm')
    plt.title('alpha l2-norm',fontsize=14)
    
    plt.subplot(2,2,3)
    term = record['delta_alpha']
    plt.scatter(range(len(term)), term, s=1)
    plt.xlabel('iteration')
    plt.ylabel('delta alpha')
    plt.title('delta alpha',fontsize=14)
    
    plt.subplot(2,2,4)
    term = [sum(x) for x in record['alpha']]
    plt.scatter(range(len(term)), term, s=1)
    plt.xlabel('iteration')
    plt.ylabel('sum alpha')
    plt.title('sum alpha',fontsize=14)
    
    fig.tight_layout()
    
    if saveFig:
        fig_title = filename.split('/')[-1]
        tit = fig.suptitle(fig_title, fontsize=16)
        fig.subplots_adjust(top=0.85)
        plt.savefig(filename+'.jpg',bbox_extra_artists=(tit,), bbox_inches='tight')

    plt.show()
'''
import math

def logRatio_dist(r_list,nbins=20,return_fig=None,rm_outlier=True,fig_size=(4,4),dpi=64):
    
    #output = para.get('output',None)
    custom_palette = sns.color_palette("Greens",5)
    
    x = np.log2(r_list)
    
    xy_lim = [math.floor(np.min(x)), math.ceil(np.max(x))] #[np.log2(1)-6, np.log2(1)+6] #[np.floor(v_min), np.floor(v_max)]
    x_m = round(np.mean(x))
    x_margin = np.max(np.array(xy_lim)-x_m)
    #xy_ticks = np.arange(xy_lim[0]+2, xy_lim[1], step=2)
    xy_ticks = np.arange(x_m-x_margin, x_m+x_margin+1, step=1)
    xy_ticks = xy_ticks[xy_ticks>0]
    
    if rm_outlier:
        x_shrinkage = rm_outliers(x)
    else:
        x_shrinkage = x
        
    #plt.figure(figsize=(6,8),dpi=256)
    fig, (ax_box,ax_hist) = plt.subplots(2, sharex=True, dpi=dpi,gridspec_kw={"height_ratios": (.1, .9)})
    fig.set_figheight(fig_size[0])
    fig.set_figwidth(fig_size[1])
    sns.boxplot(data=x, ax=ax_box, orient="h",fliersize=1,color = custom_palette[4])
    ax_box.set(xlabel='')

    xx = np.linspace(np.min(x)-1,np.max(x)+1,100)
    yy_shrinkage = scipy.stats.norm.pdf(xx,np.mean(x_shrinkage),np.std(x_shrinkage))

    a = plt.hist(x,nbins,label='trueR',alpha=0.5, color=custom_palette[4],density=True)
    top = np.max(a[0])*1.05
    plt.plot([np.mean(x_shrinkage), np.mean(x_shrinkage)], [0,top], linestyle='dashed', color='black')
    plt.plot(xx,yy_shrinkage,label='rm outliers',color='black')
    plt.xlabel('Total mRNA ratio',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.xticks(xy_ticks, np.round(2**xy_ticks,2), fontsize=12)
    #plt.yticks(np.arange(0,top+5,50),np.arange(0,top+5,50))
    #plt.suptitle(dlabel+', '+r'$e^\mu_r$='+str(round(2**np.mean(x_shrinkage),3)),fontsize=18)
    plt.suptitle(r'$R_{est}$='+str(round(2**np.mean(x_shrinkage),3)),fontsize=18)
    plt.tight_layout()
    if return_fig is True:
        return fig
    plt.show()




sns.set(style="ticks")

def get_bins(x,bw=0.05):
    return np.arange(min(x),max(x)+bw,bw)

def UMI_hist(adata,x_hist='log10UMIs',groupby=None, groups='all',return_fig=None,**fig_para):
    '''
    Visualize the log-UMI-amount distribution.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    x_hist : str, optional
        The key of logUMI values stroed in adata.obs. The default is 'log10UMIs'.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    groups : list of strings, optional
        Droplet categories to visualize logUMI distributions. It should be eithor 'all' or a list of cell type annotations specified in adata.obs[groupby].
        The default is 'all'.
    return_fig : bool, optional
        Return the matplotlib figure. The default is None.
    **fig_para : optional
         Parameters to configure the plot.

    Raises
    ------
    ValueError
        If 'groups' specifies categories not matching with adata.obs[groupby].

    Returns
    -------
    fig : matplotlib figure

    '''
    
    obsdata = adata.obs
    
    bw = fig_para.get('bw',0.05)
    fix_bins = fig_para.get('fix_bins',None)
    fix_yticks = fig_para.get('fix_yticks',None)
    palette = fig_para.get('palette','Set2')

    xticks = np.array([3,np.log10(2000),np.log10(4000),np.log10(6000),np.log10(8000),4,\
              np.log10(20000),np.log10(40000),np.log10(60000),np.log10(80000),5])
    xannos = np.array(['1k','2k','4k','6k','8k','10k','20k','40k','60k','80k','100k'])

    xidx = [i for i in range(len(xticks)) if xticks[i] > obsdata[x_hist].min() and xticks[i] < obsdata[x_hist].max()]
    
    fig = plt.figure(figsize=(6,4),dpi=64)
    ax=plt.subplot()
    
    if fix_bins is None:
        bins=get_bins(obsdata[x_hist],bw)
    else:
        bins=fix_bins   
        
    if groupby is None:
        col_list = sns.color_palette(palette, 1)
        plt.hist(obsdata[x_hist],bins,alpha=0.6, color=col_list[0])
    else:
        if groups == 'all':
            groups = obsdata[groupby].unique()
        elif isinstance(groups,list) and len([1 for v in groups if v not in obsdata[groupby].unique()]):
            raise ValueError("'show_groups' contains values absent in 'groupby'.")
            
        col_list = sns.color_palette(palette, len(groups))
        v_list = [obsdata[x_hist][obsdata[groupby]==g] for g in groups]        
        for i,g in enumerate(groups):
            plt.hist(v_list[i],bins,label=g,alpha=0.6, color=col_list[i])

        plt.legend(loc='upper right')
        
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Total cell UMI count',fontsize=12)
    plt.ylabel('Frequency',fontsize=12)
    if fix_yticks is not None:
        plt.yticks(fix_yticks,fix_yticks)
    plt.xticks(xticks[xidx], xannos[xidx])
    if return_fig is True:
        return fig



def corrected_UMI_hist(adata,groupby,groups,reference,logUMIby,ratios,return_fig=None):
    '''
    Compare raw log-UMI-amount distributions versus ratio-based corrected log-UMI-amount distributions.

    Parameters
    ----------
    adata : AnnData
        The (annotated) UMI count matrix of shape `n_obs` × `n_vars`.
        Rows correspond to droplets and columns to genes.
    groupby : str
        The key of the droplet categories stored in adata.obs. 
    groups : list, optional
        Droplet categories, e.g. ['Homo-ct1', 'Homo-ct2'] annotated in adata.obs[groupby] to display. 
        The default is None.
        We recommand putting the reference group in the first element.
    reference : str
        One droplet category annotated in adata.obs[groupby].
        Use the specified group as reference. Generally, this group has the smallest mRNA content. 
    logUMIby : str
        The key of logUMI values stroed in adata.obs. The default is 'log10UMIs'.
    ratios : list
        List of estimated total-mRNA ratios. Each element corresponds to a group specified in 'groups'. 
        We recommand setting the value of reference group to be 1.
    return_fig : bool, optional
        Return the matplotlib figure. The default is None.

    Returns
    -------
    g : matplotlib figure


    '''
    ba_info = get_corrected_logUMI(adata,groupby,groups,reference,logUMIby,ratios)
    sns.cubehelix_palette(2*len(groups), rot=-.4, light=.7)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal_list = sns.cubehelix_palette(2*len(groups), start=1.4, rot=-.25, light=.7, dark=.4) + \
    sns.cubehelix_palette(2*len(groups), rot=-.4, light=.7)
    pal = [pal_list[i] for i in [2*ii for ii in range(2*len(groups))]] 

    g = sns.FacetGrid(ba_info, row=groupby, hue=groupby, aspect=7, height=.8, palette=pal)
    g.map(sns.kdeplot, logUMIby, bw_adjust=.7, 
          cut=4, clip_on=True, fill=True, alpha=0.8, linewidth=1.5)
    g.map(sns.kdeplot, logUMIby, bw_adjust=.7, 
          cut=4, clip_on=True, color="w", lw=2)
    g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    g.map(label, groupby)
    g.fig.subplots_adjust(hspace=-0.5)
    #g.set(yticks=[], xlabel="", ylabel="",xlim=(None, 4), xticks=[3,3.5,4],title="")
    g.set(yticks=[], xlabel="", ylabel="",title="")#,xticks=xticks[xidx], xannos=xannos[xidx],title="")
    #g.set_xticklabels(xticks[xidx],xannos[xidx])
    #g.set_xticklabels(xannos[xidx])

    g.despine(bottom=True, left=True)
    #plt.savefig('./outcome/fig/UMIhist.pdf')
    plt.show()
    if return_fig is True:
        return g



def get_corrected_logUMI(adata, groupby, groups, reference,logUMIby, ratios):
    
    didx = [d for d in adata.obs_names if adata.obs.loc[d,groupby] in groups]
    #ori_data = pd.DataFrame(adata[adata.obs['danno']!='Hetero-dbl',:].obs)
    ori_data = pd.DataFrame(adata[didx,:].obs)
    ori_data[groupby] = ori_data[groupby].astype(str)
    corr_data = copy.deepcopy(ori_data)
    for i,v in enumerate(groups):
        logratio_obs = adata.uns['logUMI_para'].loc[v,'mean']-adata.uns['logUMI_para'].loc[reference,'mean']
        delta_logUMI = np.log10(ratios[i])-logratio_obs
        corr_data.loc[ori_data[groupby]==v,logUMIby] = ori_data.loc[ori_data[groupby]==v,logUMIby]+delta_logUMI#np.log10(ratio[i])
        corr_data.loc[ori_data[groupby]==v,groupby] = v+'_corrected' 
        
    corr_data[groupby] = pd.Categorical(corr_data[groupby],groups+[v+'_corrected' for v in groups])
    ori_data[groupby] = pd.Categorical(ori_data[groupby],groups)
    ba_info = pd.concat([ori_data,corr_data])

    return ba_info



def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .1, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)




with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # fxn()
    
    
    
def volcano_2DE(de_df, marker_list=None, marker_colors=None, return_fig=None):
    '''
    Plot the volcano plot of 2 DE results stored in 'de_df'.

    Parameters
    ----------
    de_df : pandas.DataFrame
        Table fusing two input DE resutls.
    markers_dn2up : list of str, optional
        Marker genes to highlight. The default is None.

    Returns
    -------
    None.

    '''
    fc_x = de_df['log2FC_gs']
    fc_y = de_df['log2FC_rc']
    genes_dn2up = de_df.index[de_df['levelchange']=='dn2up']
    genes_ns2up = de_df.index[de_df['levelchange']=='ns2up']
    genes_dn2ns =  de_df.index[de_df['levelchange']=='dn2ns']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        pval_x = - np.log10(de_df['pval_adj_gs'])
        pval_y = - np.log10(de_df['pval_adj_rc'])

    pval_x[pval_x==np.inf] = -np.log10(1e-323)
    pval_y[pval_y==np.inf] = -np.log10(1e-323)

    fig, axs =plt.subplots(1,2,figsize=(8,5),dpi=64)
    plt.subplot(1,2, 1)
    plt.scatter(fc_x,pval_x,s=0.5,color='silver')
    
    if marker_list is not None:
        for idx in range(len(marker_list)):
            plt.scatter(fc_x[marker_list[idx]], pval_x[marker_list[idx]],color=marker_colors[idx],s=20)
            
    plt.plot([0,0],[0,323],c='black',linestyle='dashed')
    plt.plot([-6,6],[-np.log10(0.05),-np.log10(0.05)],c='black',linestyle='dashed')
    plt.xticks([-5,0,5],[-5,0,5],fontsize=15) 
    plt.yticks([0,100,200,300],[0,100,200,300],fontsize=15) 
    plt.title('Global scaling DE',fontsize=18)
    plt.ylabel(r'$-log_10(p.val)$',fontsize = 18)
    plt.xlabel(r'$log_2FC$',fontsize = 18)

    plt.subplot(1,2, 2)
    plt.scatter(fc_y,pval_y,s=0.5,color='silver')

    if marker_list is not None:
        for idx in range(len(marker_list)):
            plt.scatter(fc_y[marker_list[idx]], pval_y[marker_list[idx]],color=marker_colors[idx],s=20)
            
    plt.plot([0,0],[0,323],c='black',linestyle='dashed')
    plt.plot([-6,6],[-np.log10(0.05),-np.log10(0.05)],c='black',linestyle='dashed')
    plt.xticks([-5,0,5],[-5,0,5],fontsize=15) 
    plt.yticks([0,100,200,300],[0,100,200,300],fontsize=15) 
    plt.title('RC scaling DE',fontsize=18)
    #plt.xlabel(r'$-log_10(p.val)$',fontsize = 18)
    plt.xlabel(r'$log_2FC$',fontsize = 18)
    plt.tight_layout()
    if return_fig is True:
        return fig
    plt.show()




def get_plot_df(adata, genes):

    #exp_vec = np.ravel(scipy.sparse.csr_matrix.todense(adata.raw[:,genes].X), order = 'F')
    exp_vec = np.ravel(adata[:,genes].X.toarray(), order = 'F')
    data_df = pd.DataFrame({'barcodes':adata.obs_names.tolist()*len(genes),
                            'expression': exp_vec,
                            'genes': list(itertools.chain.from_iterable([[g]*adata.n_obs for g in genes])),
                            'celltype': adata.obs['danno'].tolist()*len(genes)})
    
    return data_df




def violin_2DE(adata_1, adata_2, genes, corrected='para', data_name=['Before correction','After correction'], return_fig=None):
    '''
    Violin plot of gene expressions in global-scaling values and ratio-corrected values.

    Parameters
    ----------
    adata_1 : AnnData
        The expression matrix after global-scaling normalization.
        Rows correspond to droplets and columns to genes.
    adata_2 : AnnData
        The expression matrix after total-mRNA-ratio-based correction.
        Rows correspond to droplets and columns to genes.
    genes : list of str
        Genes to plot.
    return_fig : bool, optional
        Return the matplotlib figure. The default is None.

    Returns
    -------
    g : matplotlib figure

    '''
    data_df1 = get_plot_df(adata_1, genes)
    if 'log1p' not in adata_1.uns:
        data_df1['expression'] = np.log1p(data_df1['expression'])
    
    data_df2 = get_plot_df(adata_2, genes)
    if 'log1p' not in adata_2.uns:    
        data_df2['expression'] = np.log1p(data_df2['expression'])
    
    if 'corrected' in adata_2.uns and adata_2.uns['corrected'] == 'para':
        # shift data when displaying expression
        data_df2['expression'] = data_df2['expression'] + np.log(adata_2.uns['shift_ratio'])
    
    data_df = pd.concat([data_df1, data_df2])
    data_df['DE'] = [data_name[0]]*data_df1.shape[0] + [data_name[1]]*data_df2.shape[0]
    
    g = sns.catplot(x="genes", y="expression", hue="celltype",
                    col="DE",
                    data=data_df, kind="violin",split=True,inner = 'quartile',
                    height=4, aspect=0.15*(1+len(genes)),cut=0,bw=0.3)

    if return_fig is True:
        return g
 
    
    
 
def violin(adata,genes,return_fig=True):
    '''
    Violin of gene expression.

    Parameters
    ----------
    adata : TYPE
        DESCRIPTION.
    genes : TYPE
        DESCRIPTION.
    return_fig : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    g : TYPE
        DESCRIPTION.

    '''
    data_df = get_plot_df(adata, genes)

    g = sns.catplot(x="genes", y="expression", hue="celltype",
                    data=data_df, kind="violin",split=True,inner = 'quartile',
                    height=4, aspect=0.18*(1+len(genes)),cut=0,bw=0.3)
    
    if return_fig is True:
        return g
 
     
        
    
def DEshift(df,return_fig=None):
    
    x1 = [[df.loc['up',:].sum(), df.loc['up','up']], 
          # ns -> up
          [df.loc['up',:].sum()+df.loc['ns',:].sum(), df.loc['up','up']+df.loc['ns','up']], 
          # dn -> up
          [df.loc['up','up'],df['up'].sum()], 
          # up -> ns
          [df.loc['up',:].sum()+df.loc['ns',:].sum()+df.loc['dn','up'],df['up'].sum()+df['ns'].sum()-df.loc['dn','ns']], 
          # dn -> ns
          [df.loc['up','up']+df.loc['up','ns'], df['up'].sum()+df['ns'].sum()], 
          # up -> dn
          [df.loc['up',:].sum()+df.loc['ns','up']+df.loc['ns','ns'], df['up'].sum()+df['ns'].sum()+df.loc['up','dn']] 
          # ns -> dn
         ]
    x2 = [[df.loc['up',:].sum()+df.loc['ns','up'], df.loc['up','up']+df.loc['ns','up']],# ns -> up
          [df.loc['up',:].sum()+df.loc['ns',:].sum()+df.loc['dn','up'], df['up'].sum()],# dn -> up
          [df.loc['up','up']+df.loc['up','ns'],df['up'].sum()+df.loc['up','ns']],# up -> ns
          [df.sum().sum()-df.loc['dn','dn'],df['up'].sum()+df['ns'].sum()],# dn -> ns
          [df.loc['up',:].sum(),df['up'].sum()+df['ns'].sum()+df.loc['up','dn']],# up -> dn
          [df.loc['up',:].sum()+df.loc['ns',:].sum(),df.sum().sum()-df.loc['dn','dn']]# ns -> dn
         ]

    shading_col = ['blue','red','green','#fac748','#f88dad','#1f7a8c'] # color of DE change
    deltaDE_list = ['ns2up','dn2up','up2ns','dn2ns','up2dn','ns2dn']

    dataset = [df.sum(1).to_dict(), df.sum(0).to_dict()]
    data_orders = [['up','ns','dn'],['up','ns','dn']]

    #colors = ["#161a1d","grey","#a4161a"] # color of DE, dn ns up
    colors = ["blue","grey","red"] # color of DE, dn ns up

    names = sorted(dataset[0].keys())
    values = np.array([[data[name] for name in order] for data,order in zip(dataset, data_orders)])
    lefts = np.insert(np.cumsum(values, axis=1),0,0, axis=1)[:, :-1]
    orders = np.array(data_orders)
    bottoms = np.arange(len(data_orders))

    fig,ax = plt.subplots(figsize=(15,3))
    for name, color in zip(names, colors):
        idx = np.where(orders == name)
        value = values[idx]
        left = lefts[idx]
        plt.bar(x=left, 
                height=0.3, 
                width=value, 
                bottom=bottoms, 
                color=color, 
                alpha=0.6,
                orientation="horizontal")#, 
                #label=name)

    for i in range(6):
        if sum(np.array(x2[i]-np.array(x1[i]))) > 0:
            ax.fill_betweenx(y =[0.18,0.82], x1 = x1[i], x2 = x2[i], color = shading_col[i], label=deltaDE_list[i])
    
    up,ns,dn = df.sum(0)
    plt.text(up/2-300,0.92,up,fontsize=25)
    plt.text((up+up+ns)/2-300,0.92,ns,fontsize=25)
    plt.text((2*up+2*ns+dn)/2-300,0.92,dn,fontsize=25)

    up,ns,dn = df.sum(1)
    plt.text(up/2-300,-0.08,up,fontsize=25)
    plt.text((up+up+ns)/2-300,-0.08,ns,fontsize=25)
    plt.text((2*up+2*ns+dn)/2-300,-0.08,dn,fontsize=25)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #plt.yticks(bottoms+0.4, ["data %d" % (t+1) for t in bottoms])
    plt.yticks([])
    #plt.xticks([])
    plt.legend(loc="best", bbox_to_anchor=(1.0, 1))
    plt.subplots_adjust(right=0.85)

    plt.tight_layout()
    if return_fig is True:
        return fig
    plt.show()




