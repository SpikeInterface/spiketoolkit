from setuptools import setup, find_packages

pkg_name="spiketoolkit"

setup(
    name=pkg_name,
    version="0.1.3",
    author="Cole Hurwitz, Jeremy Magland, Alessio Paolo Buccino, Matthias Hennig",
    author_email="alessiop.buccino@gmail.com",
    description="Python toolkit for analysis, visualization, and comparison of spike sorting output",
    url="https://github.com/alejoe91/spiketoolkit",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'spikeinterface',
        'scikit-learn'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
