#!/usr/bin/env python
import numpy as np

##----------------------------------------------------------------------------- 
def H(x):
  '''
  Heaviside function
  '''
  x = np.asarray(x)
  return (x >= 0.0).astype(float)

##------------------------------------------------------------------------------
def int_H(x):
  '''
  first antiderivative of the Heaviside function
  '''
  x = np.asarray(x)
  return x*(x >= 0.0).astype(float)

##------------------------------------------------------------------------------
def int2_H(x):
  '''
  second antiderivative of the Heaviside function
  '''
  x = np.asarray(x)
  return 0.5*np.power(x,2)*(x >= 0.0).astype(float)

##------------------------------------------------------------------------------ 
def step_fun(t,a,t_o):
  '''
  step function with unit step at t_o
  '''
  return a*H(t - t_o)

##-----------------------------------------------------------------------------
def ramp_fun(t,a,t_bounds):
  '''
  ramp function which increases from 0 to 1 over the interval t_bounds[0] to 
  t_bounds[1]
  '''
  numerator = a*int_H(t - t_bounds[0]) - a*int_H(t - t_bounds[1])
  denominator = t_bounds[1] - t_bounds[0]
  return numerator/denominator

##-----------------------------------------------------------------------------
def int_step_fun(t,a,t_o):
  '''
  first antiderivative of the step function
  '''
  return a*int_H(t - t_o)

##-----------------------------------------------------------------------------
def int_ramp_fun(t,a,t_bounds):
  '''
  first antiderivative of the ramp function
  '''
  numerator = a*int2_H(t - t_bounds[0]) - a*int2_H(t - t_bounds[1])
  denominator = t_bounds[1] - t_bounds[0]
  return numerator/denominator

##-----------------------------------------------------------------------------
def diff_ramp_fun(t,a,t_bounds):
  '''
  first antiderivative of the ramp function
  '''
  numerator = a*H(t - t_bounds[0]) - a*H(t - t_bounds[1])
  denominator = t_bounds[1] - t_bounds[0]
  return numerator/denominator

##-----------------------------------------------------------------------------
def log_fun(t,a,b,t_o):
  '''
  log function
  '''
  t_shift = t - t_o
  out = np.zeros(np.shape(t_shift))
  out[t_shift > 0.0] = a*np.log(b*(t_shift[t_shift > 0.0]) + 1)
  return out

##-----------------------------------------------------------------------------
def dlog_fun_da(t,a,b,t_o):
  '''
  derivative of log function w.r.t a 
  '''
  return np.log(b*(t-t_o) + 1)

##-----------------------------------------------------------------------------
def dlog_fun_db(t,a,b,t_o):
  '''
  derivative of log function w.r.t b
  '''
  return a*(t-t_o)/(b*(t-t_o) + 1)

##-----------------------------------------------------------------------------
def int_log_fun(t,a,b,t_o):
  '''
  time integral of log function
  '''
  return -a*b*((t-t_o)/b - np.log(b**2*(t-t_o) + b)/b**2) + a*(t-t_o)*np.log(b*(t-t_o) + 1)

##-----------------------------------------------------------------------------
def dint_log_fun_da(t,a,b,t_o):
  '''
  derivative of time integral of log function w.r.t a
  '''
  return (t-t_o)*np.log(b*(t-t_o) + 1) - (t-t_o) + np.log(b**2*(t-t_o) + b)/b

##-----------------------------------------------------------------------------
def dint_log_fun_db(t,a,b,t_o):
  '''
  derivative of time integral of log function w.r.t b
  '''
  return a*(b*(t-t_o) - np.log(b*(b*(t-t_o) + 1)) + 1)/b**2

##-----------------------------------------------------------------------------
def steps_and_ramps(coseismic_times,afterslip_times):
  '''
  returns a slip function and jacobian function for a slip history described 
  by step functions and ramp functions
  '''
  coseismic_times = np.asarray(coseismic_times)
  afterslip_times = np.asarray(afterslip_times)

  c_no = len(coseismic_times)
  a_no = len(afterslip_times)
  def slipfunc(params,t,diff=0):
    '''
    returns slip at specified times using the given slip parameters

    I: number of slip parameters describing slip on each fault patch
    J: number of fault patches
    K: number of slip directions (2 unless I decide to include tensile motion)
    L: number of times where output is specified
    
    PARAMETERS
    ----------
      params: slip parameters. This can be an array of any dimension as long
              as the first dimension has length I
  
    RETURNS
    -------
      out: slip history at specified times.  This is an Lx... array where the
           ellipses represent additional dimensions that "params" might have had
   
    '''           
    t = np.asarray(t)
    #t_no = len(t)
    params = np.asarray(params)
    pshape = np.shape(params)
    assert pshape[0] == c_no + a_no
    out = np.zeros(pshape[1:])
    if diff == 1:
      for i in range(a_no):
        out += diff_ramp_fun(t,params[c_no+i,...],afterslip_times[i])

    if diff == 0:
      for i in range(c_no):
        # the None allows me to broadcast the two arrays
        out += step_fun(t,params[i,...],coseismic_times[i])

      for i in range(a_no):
        out += ramp_fun(t,params[c_no+i,...],afterslip_times[i])

    elif diff == -1:
      for i in range(c_no):
        # the None allows me to broadcast the two arrays
        out += int_step_fun(t,params[i,...],coseismic_times[i])

      for i in range(a_no):
        out += int_ramp_fun(t,params[c_no+i,...],afterslip_times[i])

    #out = np.einsum('...l->l...',out)
    return out

  def slipjac(params,t,diff=0):
    '''
    returns slip at specified times using the given slip parameters

    I: number of slip parameters describing slip on each fault patch
    J: number of fault patches
    K: number of slip directions (2 unless I decide to include tensile motion)
    L: number of times where output is specified
    
    PARAMETERS
    ----------
      params: slip parameters. This can be an array of any dimension as long
              as the first dimension has length I
  
    RETURNS
    -------
      out: derivative of slip history with respect to each of the slip 
           parameters.  This is an LxIx... array where the ellipses represent 
           additional dimensions that "params" might have had
   
    '''          
    t = np.asarray(t)
    params = np.asarray(params)
    pshape = np.shape(params)
    assert pshape[0] == c_no + a_no
    out = np.zeros(pshape)
    if diff == 0:
      for i in range(c_no):
        out[i,...] += step_fun(t,1.0,coseismic_times[i])
      for i in range(a_no):
        out[c_no+i,...] += ramp_fun(t,1.0,afterslip_times[i])
    elif diff == -1:
      for i in range(c_no):
        out[i,...] += int_step_fun(t,1.0,coseismic_times[i])
      for i in range(a_no):
        out[c_no+i,...] += int_ramp_fun(t,1.0,afterslip_times[i])

    #out = np.einsum('...l->l...',out)
    return out

  return slipfunc,slipjac

def steps_and_log(coseismic_times,t_o):
  '''
  returns a slip function and jacobian function for a slip history described 
  by step functions and log functions
  '''

  coseismic_times = np.asarray(coseismic_times)
  c_no = len(coseismic_times)
  def slipfunc(params,t,time_integral=False):
    '''
    returns slip at specified times using the given slip parameters

    I: number of slip parameters describing slip on each fault patch
    J: number of fault patches
    K: number of slip directions (2 unless I decide to include tensile motion)
    L: number of times where output is specified
    
    PARAMETERS
    ----------
      params: slip parameters. This can be an array of any dimension as long
              as the first dimension has length I
  
    RETURNS
    -------
      out: slip history at specified times.  This is an Lx... array where the
           ellipses represent additional dimensions that "params" might have had
   
    '''           
    t = np.asarray(t)
    t_no = len(t)
    params = np.asarray(params)
    pshape = np.shape(params)
    assert pshape[0] == c_no + 2
    out = np.zeros(pshape[1:]+(t_no,))
    if not time_integral:
      for i in range(c_no):
        out += step_fun(t,params[i,...,None],coseismic_times[i])
      out += log_fun(t,params[c_no,...,None],params[c_no+1,...,None],t_o)
    else:
      for i in range(c_no):
        out += int_step_fun(t,params[i,...,None],coseismic_times[i])
      out += int_log_fun(t,params[c_no,...,None],params[c_no+1,...,None],t_o)
    out = np.einsum('...l->l...',out)
    return out

  def slipjac(params,t,time_integral=False):
    '''
    returns slip at specified times using the given slip parameters

    I: number of slip parameters describing slip on each fault patch
    J: number of fault patches
    K: number of slip directions (2 unless I decide to include tensile motion)
    L: number of times where output is specified
    
    PARAMETERS
    ----------
      params: slip parameters. This can be an array of any dimension as long
              as the first dimension has length I
  
    RETURNS
    -------
      out: derivative of slip history with respect to each of the slip 
           parameters.  This is an LxIx... array where the ellipses represent 
           additional dimensions that "params" might have had   
    '''          
    t = np.asarray(t)
    t_no = len(t) 
    params = np.asarray(params)
    pshape = np.shape(params)
    assert pshape[0] == c_no + 2
    out = np.zeros(pshape+(t_no,))
    if not time_integral:
      for i in range(c_no):
        out[i,...] += step_fun(t,1.0,coseismic_times[i])
      out[c_no,...] += dlog_fun_da(t,params[c_no,...,None],params[c_no+1,...,None],t_o)
      out[c_no+1,...] += dlog_fun_db(t,params[c_no,...,None],params[c_no+1,...,None],t_o)
    else:
      for i in range(c_no):
        out[i,...] += int_step_fun(t,1.0,coseismic_times[i])
      out[c_no,...] += dint_log_fun_da(t,params[c_no,...,None],params[c_no+1,...,None],t_o)
      out[c_no+1,...] += dint_log_fun_db(t,params[c_no,...,None],params[c_no+1,...,None],t_o)

    out = np.einsum('...l->l...',out)
    return out

  return slipfunc,slipjac
