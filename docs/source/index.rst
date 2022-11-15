
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

If you want to use TOMAS to analyze your scRNA-seq data, please first install TOMAS according to the :doc:`installation` tutorial and then check out the :doc:`tutorials` section for detailed information. The :doc:`TOMAS_basic` and :doc:`Total_mRNA_aware_analyses` will introduce you the major APIs of TOMAS to infer total-mRNA ratios and perform accuarte single-cell analyses with a basic two-cell-type case. Then according to your single-cell experiment design, you could choose how to run TOMAS. If you have presort cells and perform cell hashing before single-cell library prep., follow the tutorial for :doc:`hashing_case`. If you just conduct a generic single-cell library prep., follow the tutorial :doc:`Generic_case`. 



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


   


