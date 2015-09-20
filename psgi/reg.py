#!/usr/bin/env python
from __future__ import division
import numpy as np
import sys
import modest
from modest import Perturb
import modest.misc as misc
sys.path.append('.')
import basis

def reg_points(knots):
  '''
  returns a list of collocation points used for regularization. These consists
  of the knots and the midpoints between the knots
  '''
  knots = np.array(knots,copy=True)
  midpoints = 0.5*np.diff(knots) + knots[:-1]    
  return np.concatenate((knots,midpoints))

def create_formatted_regularization(params,p):
  f = {}
  order = params['slip_regularization_order']
  if order == 0:
    N = p['Ns']
    slip_reg = params['slip_regularization']*np.eye(N)
    slip_reg = np.concatenate((slip_reg[...,None],slip_reg[...,None]),axis=-1)

  elif order == 2:
    import basis
    M = basis.FAULT_SEGMENTS
    N = basis.FAULT_N
    slip_reg = np.zeros((0,N))
    for m in range(M):
      col_x = np.unique(basis.FAULT_KNOTS[m][0])
      col_x = reg_points(col_x)
      col_y = np.unique(basis.FAULT_KNOTS[m][1])
      col_y = reg_points(col_y)
      xgrid,ygrid = np.meshgrid(col_x,col_y)
      xflat = xgrid.flatten()
      yflat = ygrid.flatten()
      zflat = 0*yflat
      col_m = np.array([xflat,yflat,zflat]).transpose()
      col_m = basis.FAULT_TRANSFORMS[m](col_m)

      slip_reg_m = np.zeros((len(col_m),N))
      for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
        slip_reg_m[:,ni] = basis.slip(col_m,n,segment=m,diff=(2,0))
        slip_reg_m[:,ni] *= params['slip_regularization']

      slip_reg = np.vstack((slip_reg,slip_reg_m))

      slip_reg_m = np.zeros((len(col_m),N))
      for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
        slip_reg_m[:,ni] = basis.slip(col_m,n,segment=m,diff=(0,2))
        slip_reg_m[:,ni] *= params['slip_regularization']
    
      slip_reg = np.vstack((slip_reg,slip_reg_m))
      slip_reg = np.concatenate((slip_reg[...,None],slip_reg[...,None]),axis=-1)

  else:
    print('invalid slip order')

  f['slip'] = slip_reg

  order = params['fluidity_regularization_order']
  if order == 0:
    N = p['Nv']
    fluidity_reg = params['fluidity_regularization']*np.eye(N)  

  elif order == 2:
    import basis
    N = basis.FLUIDITY_N
    fluidity_reg = np.zeros((0,basis.FLUIDITY_N))
    col_x = np.unique(basis.FLUIDITY_KNOTS[0])
    col_x = reg_points(col_x)
    col_y = np.unique(basis.FLUIDITY_KNOTS[1])
    col_y = reg_points(col_y)
    col_z = np.unique(basis.FLUIDITY_KNOTS[2])
    col_z = reg_points(col_z)
    xgrid,ygrid,zgrid = np.meshgrid(col_x,col_y,col_z)
    xflat = xgrid.flatten()
    yflat = ygrid.flatten()
    zflat = zgrid.flatten()
    col_m = np.array([xflat,yflat,zflat]).transpose()
    col_m = basis.FLUIDITY_TRANSFORM(col_m)
    fluidity_reg = np.zeros((len(col_m),N))
    for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
      fluidity_reg[:,ni] = basis.fluidity(col_m,n,diff=(0,0,2))
      fluidity_reg[:,ni] *= params['fluidity_regularization']

  else:
    print('invalid fluidity order')

  f['fluidity'] = fluidity_reg

  return f

def create_regularization(reg,p):
  reg_matrix = np.zeros((0,p['total']))
  for i,v in reg.iteritems():
    if len(np.shape(v)) == 3:
      for j in range(np.shape(v)[2]):
        r1 = v[:,:,j]
        r2 = np.zeros((len(r1),p['total']))
        r2[:,p[i][:,j]] = r1
        reg_matrix = np.vstack((reg_matrix,r2))

    else:
      r1 = v[:,:]
      r2 = np.zeros((len(r1),p['total']))
      r2[:,p[i]] = r1
      reg_matrix = np.vstack((reg_matrix,r2))

  return reg_matrix


  



  



