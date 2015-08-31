#!/usr/bin/env python 

# This script is used to generate the prior.h5 file which is needed as
# input for PSGI.py

# State variables that do not have a prior specified in this script will
# be assumed zero with high confidence

import numpy as np
import h5py
import argparse
import sys
sys.path.append('.')
import basis

p = argparse.ArgumentParser()
p.add_argument('--secular_velocity_variance',type=float,default=1e3)
p.add_argument('--secular_velocity_file',type=str,default=None)
p.add_argument('--baseline_displacement_variance',type=float,default=1e3)
p.add_argument('--slip_variance',type=float,default=1e-10)
p.add_argument('--fluidity_variance',type=float,default=1e3)
args = vars(p.parse_args())

Ns = basis.FAULT_N
Ds = 2
Nv = basis.FLUIDITY_N
# find the number of stations
f = h5py.File('data.h5','r')
Nx = np.shape(f['mean'])[1]
Dx = np.shape(f['mean'])[2]
f.close()

out = h5py.File('prior.h5','w')

# --------------------------------------------------------------------
# Secular Velocity for each station
# shape for mean is (Nx,Dx)
# shape for covariance is (Nx,Dx,Nx,Dx)

# set mean secular velocity prior to zero

if args['secular_velocity_file'] is not None:
  sec_vel = np.loadtxt(args['secular_velocity_file'])
   
else:
  sec_vel = np.zeros((Nx,Dx))

sec_vel_var = np.ones((Nx,Dx))
sec_vel_var *= args['secular_velocity_variance']

out['secular_velocity/mean'] = sec_vel
out['secular_velocity/variance'] = sec_vel_var

# --------------------------------------------------------------------
# Baseline displacement for each station
# shape for mean is (Nx,Dx)
# shape for covariance is (Nx,Dx,Nx,Dx)

# set mean secular velocity prior to zero
baseline = np.zeros((Nx,Dx))

baseline_var = np.ones((Nx,Dx))
baseline_var *= args['baseline_displacement_variance']

out['baseline_displacement/mean'] = baseline
out['baseline_displacement/variance'] = baseline_var

# --------------------------------------------------------------------
# Slip prior
# shape for mean is (Ns,Ds)
# shape for covariance is (Ns,Ds,Ns,Ds)

# set mean secular velocity prior to zero
slip = np.zeros((Ns,Ds))

slip_var = np.ones((Ns,Ds))
slip_var *= args['slip_variance']

out['slip/mean'] = slip
out['slip/variance'] = slip_var

# --------------------------------------------------------------------
# Fluidity prior (NOT exp(Fluidity))
# shape for mean is (Nv)
# shape for covariance is (Nv,Nv)

# set mean secular velocity prior to zero
fluidity = 1e-2*np.ones(Nv)

fluidity_var = np.ones(Nv)
fluidity_var *= args['fluidity_variance']

out['fluidity/mean'] = fluidity
out['fluidity/variance'] = fluidity_var

out.close()
