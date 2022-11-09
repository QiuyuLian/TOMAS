
.. image:: _static/img/logo_h3.png
  :width: 600
  :align: center
  
.. image:: _static/img/empty.png

TOMAS: Total-mRNA-aware single cell analysis
============================================

**TOMAS** is a Python package for **TO**\tal-**M**\RNA-**A**\ware **S**\ingle-cell analysis for droplet-based single cell sequencing. It provides a computational soluation for controlling global abundance differences in droplet-based scRNA-seq where no experimental soluations are available yet.

TOMAS is compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/#>`_ and is easy to use.


Key applications
----------------

* infer total-mRNA ratios between cell populations
* perform total-mRNA-aware sinlge-cell analyses including differential expression analysis and functional enrichment analysis
* identify potentially key genes which are moderately expressed
* visualize and compare the results with or without considering the total-mRNA differences
* correct the distortion induced by mRNA recovery rate decay 



TOMAS's contribution to your research
-------------------------------------

If you use droplet-based scRNA-seq technique in your research, TOMAS could 
 * provide a computational framework to control total mRNA content differences between cell populations;
 * empower accurate scRNA-seq analyses including differential expression analysis and functional enrichment analysis, especially for datasets with large total mRNA content differences between cell types;
 * enable large-scale, low per-cell cost scRNA-seq experiments as limiting the cell throughput on purpose to avoid doublets is no longer nesscary; 
 * improve the sensitivity and statistical power in detecting and studying rare cell types.



Getting started with TOMAS
--------------------------

If you are new to TOMAS and hope to apply it on your own data analysis, please first check out the :doc:`installation` for how to install TOMAS and the :doc:`tutorials` section for detailed information. If your data is a general scRNA-seq dataset, check out the :doc:`general_case`. If you have conducted sorting and cell hashing before single-cell library prep., check out the :doc:`hashing_case`.



Support
-------

Your feedback will be greatly welcome! Feel free to submit an `issue <https://github.com/QiuyuLian/TOMAS/issues>`_ or send us an `email <mailto:qiuyu.lian@sjtu.edu.cn>`_ if you encounter a bug when running TOMAS or if you have any suggestions about potential features TOMAS could involve. We really appreicaite your help to improve TOMAS.  



Cite us
-------

If you find it useful, please cite `our preprint <https://www.researchsquare.com/article/rs-2211167/v1>`_. 


.. toctree::
    :caption: Main
    :maxdepth: 1
    :hidden:
    
    about
    installation
    api
    contributors
    references


.. toctree::
    :caption: Tutorials
    :maxdepth: 2
    :hidden:

    TOMAS_basics
    Total_mRNA_aware_analyses
    Generic_case
    Hashing_case
    gui


   


