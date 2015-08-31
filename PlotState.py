#!/usr/bin/env python

# This script requires the following
#
import h5py
import psgi.plot_fit
import numpy as np
import argparse
import psgi.plot_state

p = argparse.ArgumentParser(
      description='plots the data and best fit data found from PSGI')

p.add_argument('--quiver_scale',type=float,default=0.01)
p.add_argument('--scale_length',type=float,default=0.1)
p.add_argument('--draw_map',type=bool,default=True)
p.add_argument('--slip_clim',type=list,default=[0.0,5.0])
p.add_argument('--fluidity_clim',type=list,default=[0.0,5.0])

param = vars(p.parse_args())

data_file = h5py.File('out.h5','r')
state = data_file['state']
psgi.plot_state.view(state,param)


