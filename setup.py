from setuptools import setup, find_packages
import unittest

pkg_name="spiketoolkit"

def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(
    name=pkg_name,
    version="0.1.0",
    author="Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig",
    author_email="alessiop.buccino@gmail.com",
    description="Python toolkit for analysis, visualization, and comparison of spike sorting output",
    url="https://github.com/alejoe91/spiketoolkit",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'spikeinterface',
        'scikit-learn',
        'ml_ms4alg',
        'spikewidgets'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    test_suite='setup.my_test_suite'
)
