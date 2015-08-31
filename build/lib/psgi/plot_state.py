#!/usr/bin/env python
from __future__ import division
from tplot.xsection import XSection
from tplot.xsection import VectorXSection
from tplot.axes3d import Axes3D
from psgi.filter import state_parser
from matplotlib.widgets import Slider
from traits.api import HasTraits, Range, Instance, on_trait_change, Array
from traitsui.api import View, Item, Group
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor,MlabSceneModel
import mayavi.mlab
import modest
import numpy as np
import pickle
import misc
import transform as trans
import sys
sys.path.append('.')
import basis

Nl = 50
Nw = 50

def slip_vec(x,coeff,strike,dip):
  s1 = basis.slip(x,coeff[:,0])
  s2 = basis.slip(x,coeff[:,1])
  vec = np.array([s1,s2,0*s2]).transpose()
  argz = np.pi/2.0 - np.pi*strike/180
  argx = np.pi/180.0*dip
  T = trans.point_rotation_x(argx)
  T += trans.point_rotation_z(argz)
  vec = T(vec)
  return vec

def slip_mag(x,coeff):
  rightlateral = basis.slip(x,coeff[:,0])
  thrust = basis.slip(x,coeff[:,1])
  return np.sqrt(rightlateral**2 + thrust**2)

def view(state,param):
  param = {i:np.array(v) for i,v in param.iteritems()}

  #covert lat lon to xyz
  f = open('basemap.pkl','r')
  bm = pickle.load(f)
  f.close()
   
  fluidity_transforms = []
  x,y = bm(*basis.FLUIDITY_ANCHOR[:2])
  length = basis.FLUIDITY_LENGTH
  width = basis.FLUIDITY_WIDTH
  thickness = basis.FLUIDITY_THICKNESS

  t = trans.point_stretch([basis.FLUIDITY_LENGTH,
                            basis.FLUIDITY_THICKNESS,
                            1.0])
  t += trans.point_rotation_x(np.pi/2.0)
  t += trans.point_translation([0.0,-width/2.0,0.0])
  t += trans.point_rotation_z(np.pi/2.0 - basis.FLUIDITY_STRIKE*np.pi/180) 
  t += trans.point_translation([x,y,0.0])
  fluidity_transforms += [t]

  t = trans.point_stretch([basis.FLUIDITY_WIDTH,
                            basis.FLUIDITY_THICKNESS,
                            1.0])
  t += trans.point_rotation_x(np.pi/2.0)
  t += trans.point_rotation_z(-np.pi/2.0)
  t += trans.point_translation([basis.FLUIDITY_LENGTH/2.0,
                                0.0,
                                0.0])
  t += trans.point_rotation_z(np.pi/2.0 - basis.FLUIDITY_STRIKE*np.pi/180) 
  t += trans.point_translation([x,y,0.0])
  fluidity_transforms += [t]

  fault_transforms = basis.FAULT_TRANSFORMS

  xs1 = XSection(basis.fluidity,
                f_args=(np.exp(state['fluidity'][-1]),),
                base_square_y=(-1,0),
                transforms = fluidity_transforms,
                 clim = param['fluidity_clim'])

  xs2 = XSection(basis.fluidity,
                f_args=(np.exp(state['fluidity'][-1]),),
                base_square_y=(-1,0),
                transforms = fault_transforms)

  class InteractiveSlip(HasTraits):
    #time_index = Range(0,len(state['slip']),0.5)
    print(state)
    time = Range(round(min(state['time']),2),round(max(state['time']),2))
    scene = Instance(MlabSceneModel,())
    view = View(Item('scene',editor=SceneEditor(scene_class=MayaviScene),
                height=250,width=300,show_label=False),
                Group('time'),resizable=True)

    def __init__(self):
      time_index = np.argmin(abs(state['time'][...] - self.time))
      slip = np.array(state['slip'][time_index])
      self.xs = ()
      self.vxs = ()
      for i,t in enumerate(fault_transforms):
        self.xs += XSection(slip_mag,
                              f_args=(slip,),
                              base_square_y=(-1,0),
                              transforms = [t],clim=param['slip_clim']),
        self.vxs += VectorXSection(slip_vec,
                                   f_args=(slip,basis.FAULT_STRIKE[i],basis.FAULT_DIP[i]),
                                   base_square_y=(-1,0),
                                   transforms = [t]),
      HasTraits.__init__(self)

    @on_trait_change('time,scene.activated')
    def update_plot(self):
      time_index = np.argmin(abs(state['time'][...] - self.time))
      slip = np.array(state['slip'][time_index])
      for i,t in enumerate(fault_transforms):
        self.xs[i].set_f_args((slip,))
        self.vxs[i].set_f_args((slip,basis.FAULT_STRIKE[i],basis.FAULT_DIP[i]))
        if self.xs[i]._plots is None:
          self.xs[i].draw()
        else:
          self.xs[i].redraw()   
        if self.vxs[i]._plots is None:
          self.vxs[i].draw()
        else:
          self.vxs[i].redraw()   

  mayavi.mlab.figure(1)
  xs1.draw()
  xs2.draw(color=(0.2,0.2,0.2),opacity=0.5)

  #mayavi.mlab.figure(2)
  xs2 = InteractiveSlip()
  xs2.configure_traits()
