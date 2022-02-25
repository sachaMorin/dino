import os
from setuptools import setup

# :==> Fill in your project data here
version = '0.0.1'
library_name = 'dt-dino'
library_webpage = 'https://github.com/sachaMorin/dino'
maintainer = 'Miguel Saavedra'
maintainer_email = 'miguel.angel.saavedra.ruiz@umontreal,ca'
short_description = 'Python library to RUN DINO segmenetation in Duckietown'
full_description = """
Python library to RUN DINO segmenetation in Duckietown.
"""
# <==: Fill in your project data here

# read project dependencies
dependencies_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
with open(dependencies_file, 'rt') as fin:
    dependencies = list(filter(lambda line: not line.startswith('#'), fin.read().splitlines()))

# compile description
underline = '=' * (len(library_name) + len(short_description) + 2)
description = """
{name}: {short}
{underline}
{long}
""".format(name=library_name, short=short_description, long=full_description, underline=underline)

# setup package
setup(name=library_name,
      author=maintainer,
      author_email=maintainer_email,
      url=library_webpage,
      install_requires=dependencies,
      packages=['dt_segmentation'],
      long_description=description,
      version=version,
      include_package_data=True)
