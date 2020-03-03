level_maker
===========

.. image:: https://img.shields.io/pypi/v/level_maker.svg
    :target: https://pypi.python.org/pypi/level_maker
    :alt: Latest PyPI version

.. image:: None.png
   :target: None
   :alt: Latest Travis CI build status

NCAR/CGD/AMP Generator for atmosphere hybrid-sigma coefficients.

Usage
-----
Provides command-line capability to produce hybrid-sigma coefficients using makelev.py.

Installation
------------
No installation requirements. Just run as command-line in an environment meeting the Requirements.

Requirements
^^^^^^^^^^^^
To run basic calculation, requires numpy and python >= 3.6 (because it uses f-strings).

To save to netCDF, requires xarray.

Tests (when functional) require pytest.

Compatibility
-------------

Licence
-------
MIT

Authors
-------

`level_maker` was written by `Brian Medeiros <brianpm@ucar.edu>`_.

Based on fortran code supplied by David L. Williamson. Documentation of algorithm in 
    David L. Williamson, Jerry G. Olson, and Byron A. Boville. 
    A comparison of semi-Lagrangian and Eulerian tropical climate simulations. 
    Monthly Weather Review, 126(4):1001–1012, 1998. 
    `doi: 10.1175/1520-0493(1998)126<1001:ACOSLA>2.0.CO;2`
