#!/usr/bin/env python
#
# This script defines variables and functions which are needed for
# plotting purposes and for using WriteRegularization.py which forms a
# regularization matrix by collocation
#
# Variables in all caps are required for other scripts to run and 
# this script must also define the slip and fluidity basis functions
#
from __future__ import division
from spectral.bspline import augmented_knots
from spectral.bspline import natural_knots
from spectral.bspline import bspline_nd
from modest import linear_to_array_index
from modest import Perturb
import transform as trans
import pickle
import numpy as np

## Define parameters for slip basis function geometry
######################################################################
FAULT_ANCHOR = [[-116.0,32.0]]
FAULT_LENGTH = [50000.0]
FAULT_WIDTH = [20000.0]
FAULT_STRIKE = [0.0]
FAULT_DIP = [60.0]
FAULT_NLENGTH = [5]
FAULT_NWIDTH = [2]
FAULT_ORDER = [[0,0]]
FLUIDITY_ANCHOR = [-119.25,35.0]
FLUIDITY_STRIKE = 90.0
FLUIDITY_LENGTH = 600000.0
FLUIDITY_WIDTH = 600000.0
FLUIDITY_THICKNESS = 150000.0
FLUIDITY_NLENGTH = 1
FLUIDITY_NWIDTH = 1
FLUIDITY_NTHICKNESS = 5
FLUIDITY_ORDER = [0,0,3]
######################################################################
 
FAULT_N = sum(l*w for l,w in zip(FAULT_NLENGTH,FAULT_NWIDTH))
FLUIDITY_N = FLUIDITY_NLENGTH*FLUIDITY_NWIDTH*FLUIDITY_NTHICKNESS

FAULT_SEGMENTS = len(FAULT_ANCHOR)
FAULT_TRANSFORMS = []
FAULT_KNOTS = []

BASEMAP = pickle.load(open('basemap.pkl','r'))

# find knots for faults
for d in range(FAULT_SEGMENTS):
  xc,yc = BASEMAP(*FAULT_ANCHOR[d])   
  t = trans.point_stretch([FAULT_LENGTH[d],FAULT_WIDTH[d],1.0])
  t += trans.point_rotation_x(FAULT_DIP[d]*np.pi/180)
  t += trans.point_rotation_z(np.pi/2.0 - FAULT_STRIKE[d]*np.pi/180)
  t += trans.point_translation([xc,yc,0.0])

  # create knots defining B-splines for slip on a rectangle x = [0,1] 
  # and y = [-1,0]
  fault_knots_x = natural_knots(FAULT_NLENGTH[d],
                                FAULT_ORDER[d][0],side='both')
  fault_knots_y = natural_knots(FAULT_NWIDTH[d],
                                FAULT_ORDER[d][1],side='both') - 1.0
  FAULT_TRANSFORMS += [t]
  FAULT_KNOTS += [(fault_knots_x,fault_knots_y)]

# find knots for fluidity
xc,yc = BASEMAP(*FLUIDITY_ANCHOR)   
t = trans.point_stretch([FLUIDITY_LENGTH,FLUIDITY_WIDTH,FLUIDITY_THICKNESS])
t += trans.point_rotation_z(np.pi/2.0 - FLUIDITY_STRIKE*np.pi/180)
t += trans.point_translation([xc,yc,0.0])

fluidity_knots_x = natural_knots(FLUIDITY_NLENGTH,
                                 FLUIDITY_ORDER[0],side='both')
fluidity_knots_y = natural_knots(FLUIDITY_NWIDTH,
                                 FLUIDITY_ORDER[1],side='both') - 1.0
fluidity_knots_z = natural_knots(FLUIDITY_NTHICKNESS,
                                 FLUIDITY_ORDER[2],side='none') - 1.0

FLUIDITY_TRANSFORM = t
FLUIDITY_KNOTS = (fluidity_knots_x,fluidity_knots_y,fluidity_knots_z)


def slip(x,coeff,segment=None,diff=None):
  '''
  takes positions, x, and slip coefficients, coeff, and returns the
  vaues for slip. The segment key word is specified to only use
  coefficients corresponding to the specified fault segment.  if no
  segment is specified then all coefficients will be used
  '''
  minN = 0
  s = segment
  out = np.zeros(len(x))
  assert len(coeff) == FAULT_N, (
    'coefficient list must have length %s' % FAULT_N)

  if s is None:
    for d in range(FAULT_SEGMENTS):
      t = FAULT_TRANSFORMS[d].inverse()
      fx = t(x)[:,[0,1]]
      shape = FAULT_NLENGTH[d],FAULT_NWIDTH[d]
      order = FAULT_ORDER[d]
      maxN = minN + np.prod(shape)
      for n in range(minN,maxN):
        idx = linear_to_array_index(n-minN,shape)
        out += coeff[n]*bspline_nd(fx,FAULT_KNOTS[d],idx,order,diff=diff)

      minN += np.prod(shape)

  else:
    for d in range(s):
      shape = FAULT_NLENGTH[d],FAULT_NWIDTH[d]
      maxN = minN + np.prod(shape)
      minN += np.prod(shape)

    shape = FAULT_NLENGTH[s],FAULT_NWIDTH[s]
    maxN = minN + np.prod(shape)
    t = FAULT_TRANSFORMS[s].inverse()
    fx = t(x)[:,[0,1]]
    order = FAULT_ORDER[s]
    for n in range(minN,maxN):
      idx = linear_to_array_index(n-minN,shape)
      out += coeff[n]*bspline_nd(fx,FAULT_KNOTS[s],idx,order,diff=diff)

    minN += np.prod(shape)

  return out

def fluidity(x,coeff,diff=None):
  out = np.zeros(len(x))
  t = FLUIDITY_TRANSFORM.inverse()
  fx = t(x)
  shape = FLUIDITY_NLENGTH,FLUIDITY_NWIDTH,FLUIDITY_NTHICKNESS
  order = FLUIDITY_ORDER
  for n in range(FLUIDITY_N):
    idx = linear_to_array_index(n,shape)
    out += coeff[n]*bspline_nd(fx,FLUIDITY_KNOTS,idx,order,diff=diff)

  return out

if __name__ == '__main__':
  from tplot.xsection import XSection
  import mayavi.mlab
  
  bm = BASEMAP

  sta_array = np.loadtxt('stations.txt',dtype=str)
  sta_pos = np.array(sta_array[:,[1,2]],dtype=float)
  sta_pos_x,sta_pos_y = bm(sta_pos[:,0],sta_pos[:,1])

    
  fluidity_transforms = []
  x,y = bm(*FLUIDITY_ANCHOR[:2])
  length = FLUIDITY_LENGTH
  width = FLUIDITY_WIDTH
  thickness = FLUIDITY_THICKNESS
  t = trans.point_stretch([FLUIDITY_LENGTH,
                           FLUIDITY_THICKNESS,
                           1.0])
  t += trans.point_rotation_x(np.pi/2.0)
  t += trans.point_translation([0.0,-width/2.0,0.0])
  t += trans.point_rotation_z(np.pi/2.0 - FLUIDITY_STRIKE*np.pi/180)
  t += trans.point_translation([x,y,0.0])
  fluidity_transforms += [t]

  t = trans.point_stretch([FLUIDITY_WIDTH,
                           FLUIDITY_THICKNESS,
                           1.0])
  t += trans.point_rotation_x(np.pi/2.0)
  t += trans.point_rotation_z(-np.pi/2.0)
  t += trans.point_translation([FLUIDITY_LENGTH/2.0,
                                0.0,
                                0.0])
  t += trans.point_rotation_z(np.pi/2.0 - FLUIDITY_STRIKE*np.pi/180)
  t += trans.point_translation([x,y,0.0])
  fluidity_transforms += [t]

  
  xs1 = XSection(fluidity,
                f_args=(np.random.random(FLUIDITY_N),),
                base_square_y=(-1,0),
                transforms = fluidity_transforms,
                 clim = (0,1))

  xs2 = XSection(fluidity,
                 f_args=(np.random.random(FLUIDITY_N),),
                 base_square_y=(-1,0),
                 transforms = FAULT_TRANSFORMS)
  xs1.draw()
  xs2.draw(color=(0.2,0.2,0.2),opacity=0.5)
  mayavi.mlab.points3d(sta_pos_x,sta_pos_y,0*sta_pos[:,1],scale_factor=10000)
  xs1.view()

  coeff = np.random.random(FAULT_N)
  xs1 = XSection(slip,
                 f_args=(coeff,),
                 base_square_y=(-1,0),
                 transforms = FAULT_TRANSFORMS,
                 clim=(0,1))
  xs1.draw()
  xs1.view()

  coeff = np.random.random(FLUIDITY_N)

