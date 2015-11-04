#!/usr/bin/env python
from __future__ import division
import numpy as np
import sys
import modest
from modest import Perturb
import modest.misc as misc
sys.path.append('.')
import basis

def midspace(a,b,N):
  l = np.linspace(a,b,N+1)
  return np.diff(l)/2.0 + l[:-1]

def reg_points(knots):
  '''
  returns a list of collocation points used for regularization. These consists
  of the knots and the midpoints between the knots
  '''
  knots = np.array(knots,copy=True)
  midpoints = 0.5*np.diff(knots) + knots[:-1]    
  #return midpoints
  return np.concatenate((knots,midpoints))

def _create_formatted_regularization(params,p,slip_model='stochastic'):
  f = {}
  order = params['slip_regularization_order']
  if order == 0:
    N = p['Ns']
    slip_reg = params['slip_regularization']*np.eye(N)

  elif order == 2:
    import basis
    M = basis.FAULT_SEGMENTS
    N = basis.FAULT_N
    slip_reg = np.zeros((0,N))
    for m in range(M):
      col_x = np.unique(basis.FAULT_KNOTS[m][0])
      col_x = reg_points(col_x)
      #col_x = midspace(0,1,basis.FAULT_NLENGTH[m])
      col_y = np.unique(basis.FAULT_KNOTS[m][1])
      col_y = reg_points(col_y)
      #col_y = midspace(-1,0,basis.FAULT_NWIDTH[m])
      xgrid,ygrid = np.meshgrid(col_x,col_y)
      xflat = xgrid.flatten()
      yflat = ygrid.flatten()
      zflat = 0*yflat
      col_m = np.array([xflat,yflat,zflat]).transpose()
      col_m = basis.FAULT_TRANSFORMS[m](col_m)

      slip_reg_m = np.zeros((len(col_m),N))
      for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
        slip_reg_m[:,ni] = (basis.slip(col_m,n,segment=m,diff=(2,0)) + 
                             basis.slip(col_m,n,segment=m,diff=(0,2)))
      #slip_reg_m2 = np.zeros((len(col_m),N))
      #for ni,n in enumerate(Perturb(np.zeros(N),1.0)):
      #  slip_reg_m2[:,ni] = basis.slip(col_m,n,segment=m,diff=(0,2))
    
      #slip_reg_m = np.vstack((slip_reg_m1,slip_reg_m2))
      slip_reg = np.vstack((slip_reg,slip_reg_m))
  
    slip_reg = np.concatenate((slip_reg[...,None],slip_reg[...,None]),axis=-1)
    slip_reg /= np.max(np.max(np.abs(slip_reg)))
    slip_reg *= params['slip_regularization']

  else:
    print('invalid slip order')

  if slip_model == 'parameterized':
    slip_reg = np.repeat(slip_reg[None,...],p['Nt'],axis=0)
    slip_reg = np.repeat(slip_reg[...,None],p['Ds'],axis=-1)

  if slip_model == 'stochastic':
    slip_reg = np.repeat(slip_reg[...,None],p['Ds'],axis=-1)

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

    fluidity_reg /= np.max(np.max(np.abs(fluidity_reg)))
    fluidity_reg *= params['fluidity_regularization']
  else:
    print('invalid fluidity order')

  f['fluidity'] = fluidity_reg

  return f


def _create_regularization(params,p,slip_model='stochastic'):
  reg = create_formatted_regularization(params,p,slip_model=slip_model)
  if slip_model == 'stochastic':
    reg_matrix = np.zeros((0,p['total']))
    for i,v in reg.iteritems():
      if i == 'slip':
        for j in range(p['Ds']):
          r1 = v[:,:,j]
          r2 = np.zeros((len(r1),p['total']))
          r2[:,p[i][:,j]] = r1
          reg_matrix = np.vstack((reg_matrix,r2))

      else:
        r1 = v[:,:]
        r2 = np.zeros((len(r1),p['total']))
        r2[:,p[i]] = r1
        reg_matrix = np.vstack((reg_matrix,r2))

  if slip_model == 'parameterized':
    reg_matrix = np.zeros((0,p['total']))
    for i,v in reg.iteritems():
      if i == 'slip':
        for t in range(p['Nt']):
          for j in range(p['Ds']):
            r1 = v[t,:,:,j]
            r2 = np.zeros((len(r1),p['total']))
            r2[:,p[i][t,:,j]] = r1
            reg_matrix = np.vstack((reg_matrix,r2))

      else:
        r1 = v[:,:]
        r2 = np.zeros((len(r1),p['total']))
        r2[:,p[i]] = r1
        reg_matrix = np.vstack((reg_matrix,r2))

  return reg_matrix

def create_regularization(params,p,slip_model='stochastic'):
  Nlength = basis.FAULT_NLENGTH
  Nwidth = basis.FAULT_NWIDTH
  slipC = np.zeros((0,0),dtype=int)
  count = 0
  reg = np.zeros((0,p['total']))

  # form connectivity matrix for slip
  for i,(l,w) in enumerate(zip(Nlength,Nwidth)):
    Ci = np.arange(count,count+l*w).reshape((l,w))
    Ci = Ci[::-1,:]
    if i == (len(Nlength)-1):
      slipC = modest.pad(slipC,(slipC.shape[0]+1,slipC.shape[1]),value=-1)

    slipC = modest.pad_stack((slipC,Ci),value=-1,axis=0)
    count += l*w

  import matplotlib.pyplot as plt
  plt.pcolor(slipC)
  plt.show()  
  # append slip regularization matrices to main reg matrix
  for j in range(p['Ds']):
    for i in range(p['Nst']):
      slip_indices = p['slip'][i,:,j]
      slip_indices = np.hstack((slip_indices,-1))
      con = slip_indices[slipC]
      regi = modest.tikhonov.tikhonov_matrix(
               con,
               params['slip_regularization_order'],
               column_no=p['total'])

      regi *= params['slip_regularization']
      reg = np.vstack((reg,regi))

  # form fluidity connectivity  
  Nthickness = basis.FLUIDITY_NTHICKNESS
  fluC = np.arange(Nthickness,dtype=int)
  con = p['fluidity'][fluC] 
  # form fluidity regularization
  regi = modest.tikhonov.tikhonov_matrix(
           con,
           params['fluidity_regularization_order'],
           column_no=p['total'])

  regi *= params['fluidity_regularization']
  # append to main
  reg = np.vstack((reg,regi))

  return reg



  



  



