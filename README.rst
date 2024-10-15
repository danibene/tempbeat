.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/tempbeat.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/tempbeat
    .. image:: https://img.shields.io/conda/vn/conda-forge/tempbeat.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/tempbeat
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/tempbeat

.. image:: https://img.shields.io/pypi/v/tempbeat.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/tempbeat/
.. image:: https://pepy.tech/badge/tempbeat/month
    :alt: Monthly Downloads
    :target: https://pepy.tech/project/tempbeat
.. image:: https://readthedocs.org/projects/tempbeat/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://tempbeat.readthedocs.io/en/stable/
.. image:: https://img.shields.io/coveralls/github/danibene/tempbeat/main.svg
    :alt: Coveralls
    :target: https://coveralls.io/r/danibene/tempbeat
.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

========
tempbeat
========


    Template-based interbeat interval extraction


Introduction
============
This is a Python package for extracting interbeat intervals from various heartbeat signals. It includes a template-based method developed for in-ear heartbeat sounds, which has also been tested for electrocardiography and photoplethysmography signals. The tests on in-ear heartbeat sounds are described in the following paper:

    Benesch, D., Chabot, P., Tom, A., Voix, J., & Bouserhal, R. E. (2024). Template-based Extraction of Interbeat Intervals from In-Ear Heartbeat Sounds. IEEE International Conference on Wearable and Implantable Body Sensor Networks (BSN 2024).


Installation
============
You can install the package via pip::

    pip install tempbeat

Or you can install the package from the source code::

        git clone https://github.com/danibene/tempbeat.git
        cd tempbeat
        pip install -e .

Usage
==========
.. code-block:: python

    from tempbeat.extraction.heartbeat_extraction import hb_extract
    # sig is a 1D numpy array
    # peak_time is a 1D numpy array with the time of each heartbeat in seconds
    peak_time = hb_extract(sig, sampling_rate=sampling_rate, method="temp")

To use a method implemented in MATLAB, you need to have MATLAB installed and
`the MATLAB engine for Python`_. After putting the MATLAB code in the
``src/matlab`` folder, you can use it as follows:

.. _the MATLAB engine for Python: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html


.. code-block:: python

    peak_time = hb_extract(sig, sampling_rate=sampling_rate, method="matlab")

.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd tempbeat
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

.. _pre-commit: https://pre-commit.com/

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
