
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
* perform total-mRNA-aware sinlge-cell analyses including differential expression (DE) analysis and functional enrichment analysis
* visualize and compare the results with or without considering the total-mRNA differences
* correct the distortion of scRNA-seq data induced by mRNA recovery rate decay 
* identify potential marker genes which are expressed at moderate level and thus more sensitive to total mRNA differences



Getting started with TOMAS
--------------------------

To utilize TOMAS in your single-cell studies, kindly install the package following the :doc:`installation` tutorial. Once installed, you can refer to the tutorials for guidance on how to use it. You could get the key idea of TOMAS with a basic usage scenario illustrated in :doc:`TOMAS_basics`. Depending on how you prepared your single-cell library, you could select the appropriate tutorial to follow. If you prepared your scRNA-seq library without any experimental sample multiplexing techniques (Cell-hashing, MULTI-seq, etc.), please refer to the tutorial :doc:`Generic_case`. If you adopted sample multiplexing methods during lib preparation, please follow the tutorial :doc:`Sample_Multiplexing` (still under construction). Exmaple datasets used in tutorials could be downloaded `here <https://github.com/QiuyuLian/TOMAS/tree/main/datasets>`_. 

After recovering the accurate total mRNA ratios between cell types, please refer to the tutorial :doc:`Total_mRNA_aware_analyses` for downstream analyses that consider the total mRNA differences during comparative analysis.

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
    Generic_case
    Hashing_case
    Total_mRNA_aware_analyses
    gui


   


