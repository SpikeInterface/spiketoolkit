from setuptools import setup, find_packages

d = {}
exec(open("spiketoolkit/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spiketoolkit"

setup(
    name=pkg_name,
    version=version,
    author="Alessio Buccino, Cole Hurwitz, Samuel Garcia, Jeremy Magland, Matthias Hennig",
    author_email="alessiop.buccino@gmail.com",
    description="Python toolkit for analysis, visualization, and comparison of spike sorting output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejoe91/spiketoolkit",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'spikeextractors>=0.9.4',
        'spikemetrics>=0.2.4',
        'spikefeatures',
        'scikit-learn',
        'scipy',
        'pandas',
        'networkx',
        'joblib'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
