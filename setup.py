from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bayesian_horseshoe",
    version="0.1.0",
    description="A Python package for Bayesian variable selection using Horseshoe Priors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="hyun wook sim",  
    author_email="jay.sim@mail.utoronto.ca",  
    url="https://github.com/yourusername/BayesianHorseshoe",  
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pymc3>=3.11.0",
        "arviz>=0.11.0",
        "matplotlib>=3.1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)