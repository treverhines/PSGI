#!/home/hinest/anaconda/bin/python

# This script requires the following
#
import h5py
import psgi.plot_fit
import numpy as np
import argparse

p = argparse.ArgumentParser(
      description='plots the data and best fit data found from PSGI')

p.add_argument('--quiver_scale',type=float,default=0.00001)
p.add_argument('--scale_length',type=float,default=1.0)
p.add_argument('--draw_map',type=bool,default=True)

param = vars(p.parse_args())

data_file = h5py.File('out.h5','r')
data = data_file['data']
pred = data_file['predicted']
psgi.plot_fit.view(data,pred,param)


