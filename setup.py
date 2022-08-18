import os
from setuptools import setup, find_packages
import subprocess
import logging

PACKAGE_NAME = 'sku'


setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    description='Additions to the Scikit-Learn Pipeline that is helpful for me, and maybe someone else!',
    author='Alexander Capstick',
    author_email='',
    packages=find_packages(),
    long_description=open('README.txt').read(),
    install_requires=[
                        "numpy>=1.22",
                        "scikit-learn>=1.0",
                        "joblib>=1.1.0",
                        "scikit-optimize>=0.8.1"
    ]
)