# setup.py
from setuptools import setup, find_packages

setup(
    name="atlas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "colorama>=0.4.4",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
    ],
    author="Your Name",
    description="Change Detection using Enhanced SNUNet",
    python_requires=">=3.12",
)
