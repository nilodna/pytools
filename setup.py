#!/usr/bin/env python

from setuptools import setup, find_packages

def version():
    with open('VERSION') as f:
        return f.read().strip()

def readme():
    with open('README.md') as f:
        return f.read()

reqs = [line.strip() for line in open('requirements.txt') if not line.startswith('#')]

setup(name              = "pytools",
      version           = version(),
      description       = "A package created by Danilo A. Silva to work on a daily basis",
      long_description  = readme(),
      license           = '',
      author            = 'Danilo A. Silva',
      author_email      = 'nilodna@gmail.com',
      packages          = find_packages(),
      # install_requires  = reqs,
      python_requires='>=3.6',
     )
