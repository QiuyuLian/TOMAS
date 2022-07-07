#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 02:29:17 2022

@author: qy
"""

from matplotlib import pyplot as plt
import numpy as np


    
def AlphaOptProcess(record, saveFig = False, filename=None):
    '''
    Visualize the optimation process of alpha.

    Parameters
    ----------
    record : dict
        intermediate variables saved by Function: estiamteAlpha.
    saveFig : bool, optional
        save the plot or not. The default is False.
    filename : str, optional
        path and name to save the plot. The default is None.
        its form should be '/path_to_save_the_file/name_of_the_file' without suffix.
        note: it must be specified if saveFig is True.
        
    Returns
    -------
    None.

    '''
    
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
    
    



import seaborn as sns
from matplotlib import pyplot as plt

def LogRdist(r_list,nbins=20,return_fig=None):
    
    #output = para.get('output',None)
    custom_palette = sns.color_palette("Greens",5)
    
    xy_lim = [np.log2(1)-6, np.log2(1)+6] #[np.floor(v_min), np.floor(v_max)]
    xy_ticks = np.arange(xy_lim[0]+2, xy_lim[1], step=2)

    x = np.log2(r_list)
    x_shrinkage = rm_outliers(x)

    plt.figure(figsize=(6,8),dpi=256)
    fig, (ax_box,ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.1, .9)})

    sns.boxplot(data=x, ax=ax_box, orient="h",fliersize=1,color = custom_palette[4])
    ax_box.set(xlabel='')

    xx = np.linspace(np.min(x)-1,np.max(x)+1,100)
    yy_shrinkage = stats.norm.pdf(xx,np.mean(x_shrinkage),np.std(x_shrinkage))

    a = plt.hist(x,nbins,label='trueR',alpha=0.5, color=custom_palette[4],density=True)
    top = np.ceil( np.max(a[0])/5 )*5
    plt.plot([np.mean(x_shrinkage), np.mean(x_shrinkage)], [0,0.3], linestyle='dashed', color='black')
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
    
from scipy import stats

def rm_outliers(x):
    iqr = stats.iqr(x)
    outlier_lb = np.quantile(x,0.25)-1.5*iqr
    outlier_ub = np.quantile(x,0.75)+1.5*iqr
    x_shrinkage = x[x > outlier_lb]
    x_shrinkage = x_shrinkage[x_shrinkage<outlier_ub]
    return x_shrinkage#,(outlier_lb,outlier_ub)



