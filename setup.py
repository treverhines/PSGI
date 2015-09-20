#!/usr/bin/env python
from distutils.core import setup
setup(name='PSGI',
      version='0.1',
      description='PostSeismic Geodetic Inversion',
      author='Trever Hines',
      author_email='treverhines@gmail.com',
      url='www.github.com/treverhines/PSGI',
      packages=['psgi'],
      scripts=['PSGI.py','PlotFit.py','PlotState.py'],
      license='MIT')
