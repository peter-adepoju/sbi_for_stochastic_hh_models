# setup.py

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="mf_npe",
    version="0.1.0",
    description="Multifidelity Simulation-Based Inference for Stochastic Neuron Models",
    
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    
    author="Peter Oluwafemi Adepoju",
    author_email="petera@aims.ac.za",
    
    url="https://github.com/peter-adepoju/AIMS-Final-project",
    
    # find_packages() automatically discovers and includes all packages
    # (folders with an __init__.py file) in the project.
 
    packages=find_packages(),
    
    install_requires=requirements,
    
    # Specifies the minimum Python version required.
    python_requires=">=3.10",
    
    # Additional metadata for package searching and classification.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)