API
===

Import TOMAS as tm:

>>> import TOMAS as tm


.. module:: tomas



Fitting
-------

Fitting Dirichlet-Multinomial(DMN) models and logNormal distributions of single-cell UMI counts. 

.. autosummary::
   :toctree: generated/

   tomas.fit.dmn
   tomas.fit.logN_para



Inference
---------

Inferring heterotypic doublets and the total-mRNA ratios between two cell types.

.. autosummary::
   :toctree: generated/
   
   tomas.infer.get_dbl_mg
   tomas.infer.get_dbl_mg_bc
   tomas.infer.ratios_bc
   tomas.infer.heteroDbl
   tomas.infer.heteroDbl_bc



Total-mRNA-ratio-aware analyses
-------------------------------

Performing total-mRNA-ratio-aware analyses including differential expression (DE) analysis and gene set enrichment (GSEA) analysis (under construction). 

.. autosummary::
   :toctree: generated/

   tomas.lrt.total_mRNA_aware_DE
   tomas.lrt.extract_DE
   tomas.lrt.summarize2DE



Visualization
-------------

.. autosummary::
   :toctree: generated/

   tomas.vis.UMI_hist
   tomas.vis.dmn_convergence
   tomas.vis.corrected_UMI_hist
   tomas.vis.volcano_2DE
   tomas.vis.violin_2DE


Auxiliary functions
-------------------

.. autosummary::
   :toctree: generated/

   tomas.auxi.correctUMI
   tomas.auxi.extract_specific_genes



