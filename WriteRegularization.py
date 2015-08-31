#!/usr/bin/env python
from __future__ import division
import numpy as np
import argparse
import sys
import h5py
from modest import Perturb
sys.path.append('.')
import basis

p = argparse.ArgumentParser()
p.add_argument('--fluidity',type=float,default=0.0)
p.add_argument('--slip',type=float,default=0.0)

args = vars(p.parse_args())
f = h5py.File('regularization.h5','w')

def midspace(a,b,N):
  '''                                 
  divides the range a-b into N equal length bins and then returns               
  the midpoints of the bins                             
  '''
  l = np.linspace(a,b,N+1)
  return np.diff(l)/2.0 + l[:-1]

# form slip regularization
M = basis.FAULT_SEGMENTS
N = basis.FAULT_N
slip_reg = np.zeros((0,basis.FAULT_N))
'''
for m in range(M):
  col_x = midspace(0,1,basis.FAULT_NLENGTH[m])
  col_y = midspace(-1,0,basis.FAULT_NWIDTH[m])
  xgrid,ygrid = np.meshgrid(col_x,col_y)
  xflat = xgrid.flatten()
  yflat = ygrid.flatten()
  zflat = 0*yflat
  col_m = np.array([xflat,yflat,zflat]).transpose()
  col_m = basis.FAULT_TRANSFORMS[m](col_m)

  slip_reg_m = np.zeros((len(col_m),N))
  for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
    slip_reg_m[:,ni] = args['slip']*basis.slip(col_m,n,segment=m,diff=(2,0))

  slip_reg = np.vstack((slip_reg,slip_reg_m))

  slip_reg_m = np.zeros((len(col_m),N))
  for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
    slip_reg_m[:,ni] = args['slip']*basis.slip(col_m,n,segment=m,diff=(0,2))
  
  slip_reg = np.vstack((slip_reg,slip_reg_m))

'''
slip_reg = args['slip']*np.eye(basis.FAULT_N)
slip_reg = np.concatenate((slip_reg[...,None],slip_reg[...,None]),axis=-1)
f['slip'] = slip_reg

N = basis.FLUIDITY_N
fluidity_reg = np.zeros((0,basis.FLUIDITY_N))

col_x = midspace(0,1,basis.FLUIDITY_NLENGTH)
col_y = midspace(-1,0,basis.FLUIDITY_NWIDTH)
col_z = midspace(-1,0,basis.FLUIDITY_NTHICKNESS)
xgrid,ygrid,zgrid = np.meshgrid(col_x,col_y,col_z)
xflat = xgrid.flatten()
yflat = ygrid.flatten()
zflat = zgrid.flatten()
col_m = np.array([xflat,yflat,zflat]).transpose()
col_m = basis.FLUIDITY_TRANSFORM(col_m)
fluidity_reg_m = np.zeros((len(col_m),N))
for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
  fluidity_reg_m[:,ni] = args['fluidity']*basis.fluidity(col_m,n,diff=(0,0,2))

fluidity_reg = np.vstack((fluidity_reg,fluidity_reg_m))

f['fluidity'] = fluidity_reg
f.close()


  



  



