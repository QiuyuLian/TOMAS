"""
Lumache - Python library for cooks and food lovers.
"""
import numpy as np
from scipy.special import loggamma, gammaln


def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]





def initialize_alpha(Y=1):
    """
    test.

    :param Y: Optional "kind" of ingredients.
    :type Y: list[str] or None
    :return: 10
    :rtype: int
    """
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
    
    return 10




def lgammaVec(inVec):
    """
    test import numpy.

    :param inVec: Optional "kind" of ingredients.
    :type inVec: list[str] or None
    :return: 1
    :rtype: list(float)
    """
    
    outVec = gammaln(inVec)
    idx_inf = np.isinf(outVec)
    outVec[idx_inf] = 709.1962086421661
    # lnGamma(1e-308) = 709.1962086421661, lnGamma(1e-309) = inf
    # if val is closer than 1e-308 to 0, e.g. 1e-309, inf will be returned 
    # we would truncate it to a certain number: here it's lnGamma(1e-308)
    return outVec

