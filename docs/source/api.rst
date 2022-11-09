API
===

Import TOMAS as tm:

>>> import TOMAS as tm


.. module:: tomas



Model fitting
-------------

Fitting Dirichlet-Multinomial(DMN) models with UMI counts of a droplet population. 

.. autosummary::
   :toctree: generated/

   tomas.fit.dmn
   tomas.fit.logN_para



Total-mRNA-ratio inference
--------------------------

Inferring the total-mRNA-ratio between two cell types.

.. autosummary::
   :toctree: generated/
   
   tomas.infer.ratio_2types


Total-mRNA-ratio-aware DE
-------------------------

Performing total-mRNA-ratio-aware differentail analysis.

.. autosummary::
   :toctree: generated/

   tomas.lrt.total_mRNA_aware_DE
   tomas.lrt.extract_DE
   tomas.lrt.summarize2DE


Multiple-cell-type and scRNA-only scenario
------------------------------------------

Using computational methods to perform unsupervised cell type identification and heterotypic doublet refinement.

.. autosummary::
   :toctree: generated/

   tomas.auxi.extract_specific_genes
   tomas.auxi.inferHeteroDbl


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

   tomas.auxi.cal_KL_bc
   tomas.auxi.get_dbl_mg_bc
   tomas.auxi.correctUMI



