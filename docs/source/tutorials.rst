Tutorials
=====

Basic condition
----------------

To retrieve a list of random ingredients,
you can use the ``TOMAS.get_random_ingredients()`` function:

.. autofunction:: TOMAS.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: TOMAS.InvalidKindError

For example:

>>> import TOMAS as tm
>>> tm.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']



Multiple cell types
----------------

xxx.



Only scRNA-seq data
----------------

xxx.



Simulation
----------------

xxx.



Run with Pantheon
----------------

xxx.




