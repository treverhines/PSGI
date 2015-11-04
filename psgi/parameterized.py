#!/usr/bin/env python
from __future__ import division
import numpy as np
import modest
import logging
import h5py
import scipy
import prior
import reg
from slip import steps_and_ramps
from modest import funtime

logger = logging.getLogger(__name__)

def state_parser(Nst,Ns,Ds,Nv,Nx,Dx):
  out = {}
  out['Ns'] = Ns # number of slip basis functions
  out['Ds'] = Ds # number of slip directions
  out['Nst'] = Nst # number of slip time coefficients
  out['Nv'] = Nv # number of viscous basis functions
  out['Nx'] = Nx # number of stations
  out['Dx'] = Dx # number of station displacement directions
  out['total'] = Nst*Ns*Ds + Nv + 2*Nx*Dx

  slip = np.arange(0,Nst*Ns*Ds,dtype=int)
  out['slip'] = np.reshape(slip,(Nst,Ns,Ds))

  out['fluidity'] = np.arange(Ns*Ds*Nst,Ns*Ds*Nst + Nv,dtype=int)

  secular_velocity = np.arange(Ns*Ds*Nst + Nv,Ns*Ds*Nst + Nv + Nx*Dx,dtype=int)
  out['secular_velocity'] = np.reshape(secular_velocity,(Nx,Dx))

  baseline = np.arange(Ns*Ds*Nst + Nv + Nx*Dx,Ns*Ds*Nst + Nv + 2*Nx*Dx,dtype=int)
  out['baseline_displacement'] = np.reshape(baseline,(Nx,Dx))

  return out

def flat_data(u):
  Nt,Nx,Dx = np.shape(u)
  u_flat = np.reshape(u,(Nt*Nx*Dx))   
  return u_flat

def flat_cov(cov):
  Nt,Nx,Dx,Dx = np.shape(cov) 
  cov = np.reshape(cov,(Nt*Nx,Dx,Dx))
  cov_flat = scipy.linalg.block_diag(*cov)
  return cov_flat


@funtime
def observation_t(X,t,F,G,p,slip_func,flatten=True):
  '''
  X: state vector
  t: time scalar
  F: Ns,Ds,Nx,Dx
  G: Ns,Ds,Nv,Nx,Dx
  '''
  b = slip_func(X[p['slip']],t)
  b_int = slip_func(X[p['slip']],t,diff=-1)
  slip = np.einsum('ij,ijkl',b,F)
  visc = np.einsum('ij,k,ijklm',b_int,X[p['fluidity']],G)
  tect = (X[p['secular_velocity']]*t + X[p['baseline_displacement']])
  out = tect + slip + visc
  if flatten:
    out = flat_data(out)

  return out

def observation(X,t_list,F,G,p,slip_func,flatten=True):
  out = np.zeros((len(t_list),p['Nx'],p['Dx']))
  for i,t in enumerate(t_list):
    out[i,:,:] = observation_t(X,t,F,G,p,slip_func,flatten=False)

  if flatten:
    out = flat_data(out)

  return out

@funtime
def observation_jacobian_t(X,t,F,G,p,slip_func,slip_jac):
  F = np.einsum('ijkl->klij',F)
  # change G from slip_patch,slip_dir,visc,disp,disp_dir to 
  # disp,disp_dir,visc,slip_patch,slip_dir
  G = np.einsum('ijklm->lmkij',G)

  b_int = slip_func(X[p['slip']],t,diff=-1)

  b_jac = slip_jac(X[p['slip']],t)

  b_int_jac = slip_jac(X[p['slip']],t,diff=-1)

  jac = np.zeros((p['Nx'],p['Dx'],p['total']))
  jac_vel = t*np.eye(p['Dx']*p['Nx']).reshape(p['Nx'],p['Dx'],p['Nx'],p['Dx'])
  jac_disp = np.eye(p['Dx']*p['Nx']).reshape(p['Nx'],p['Dx'],p['Nx'],p['Dx'])
  jac[:,:,p['secular_velocity']] = jac_vel
  jac[:,:,p['baseline_displacement']] = jac_disp
  jac[:,:,p['slip']] = np.einsum('ij...,...->ij...',F,b_jac)
  jac[:,:,p['slip']] += np.einsum('ijk...,k,...->ij...',G,X[p['fluidity']],b_int_jac)
  jac[:,:,p['fluidity']] = np.einsum('ij...lm,lm->ij...',
                                     G,b_int)

  jac = np.reshape(jac,(p['Nx']*p['Dx'],p['total']))   
  return jac


def observation_jacobian(X,t_list,F,G,p,slip_func,slip_jac):
  out = np.zeros((len(t_list),p['Nx']*p['Dx'],p['total']))
  for i,t in enumerate(t_list):
    out[i,:,:] = observation_jacobian_t(X,t,F,G,p,slip_func,slip_jac)

  out = out.reshape((len(t_list)*p['Nx']*p['Dx'],p['total']))
  return out


def regularization(X,reg_matrix):
  out = reg_matrix.dot(X)
  return out


def regularization_jacobian(X,reg_matrix):
  return np.array(reg_matrix,copy=True)


def L2(residual,covariance,mask):
  residual = flat_data([residual[~mask,:]])
  covariance = flat_data([covariance[~mask,:]])
  #covariance = flat_cov([covariance[~mask,:,:]])
  return np.sum(residual**2/covariance)
  #covinv = np.linalg.inv(covariance)
  #L2 = np.einsum('i,ij,j',residual,covinv,residual)
  #return L2


@funtime
def main(data,gf,param,outfile):
  '''
  Nt: number of time steps
  Nx: number of positions
  Dx: spatial dimensions of coordinates and displacements 
  Ns: number of slip basis functions per slip direction
  Ds: number of slip directions
  Nv: number of fluidity basis functions
  total: number of state parameters (Ns*Ds + Nv + 2*Nx*Dx)

  Parameters
  ----------
    data: \ mask : (Nt,Nx) boolean array
          \ mean : (Nt,Nx,Dx) array
          \ covariance : (Nt,Nx,Dx,Dx) array
          \ metadata \ position : (Nx,Dx) array
                       time : (Nt,) array

    prior: \ mean : (total,) array
           \ covariance : (total,total) array

    gf: \ elastic : (Ns,Ds,Nx,Dx) array
        \ viscoelastic : (Ns,Ds,Dv,Nx,Dx) array
        \ metadata \ position : (Nx,Dx) array

    reg: \ regularization : (*,total) array

    params: user parameter dictionary

  Returns
  -------
    out: \ slip_integral \ mean : (Nt,Ns,Ds) array
                         \ uncertainty :(Nt,Ns,Ds) array
         \ slip \ mean : (Nt,Ns,Ds) array
                \ uncertainty : (Nt,Ns,Ds) array
         \ slip_derivative \ mean : (Nt,Ns,Ds) array
                           \ uncertainty : (Nt,Ns,Ds) array
         \ fluidity \ mean : (Nv,) array
                    \ uncertainty : (Nv,) array
         \ secular_velocity \ mean : (Nx,Dx) array
                            \ uncertainty : (Nx,Dx) array
         \ baseline_displacement \ mean : (Nx,Dx) array
                                 \ uncertainty : (Nx,Dx) array

  '''
  F = gf['slip'][...]
  G = gf['fluidity'][...]
  
  coseismic_times = np.array(param['coseismic_times'])
  afterslip_start_times = param['afterslip_start_times']
  afterslip_end_times = param['afterslip_end_times']
  afterslip_times = zip(afterslip_start_times,afterslip_end_times) 
  afterslip_times = np.array(afterslip_times)
  time = data['time'][:]

  time_shifted = time - np.min(time)
  coseismic_times_shifted = coseismic_times - np.min(time)
  afterslip_times_shifted = afterslip_times - np.min(time)

  slip_func,slip_jac = steps_and_ramps(coseismic_times_shifted,
                                       afterslip_times_shifted)

  # define slip functions and slip jacobian here

  Nst = len(coseismic_times) + len(afterslip_start_times)
  Ns,Ds,Nv,Nx,Dx = np.shape(G)
  Nt = len(time)
  
  # check for consistency between input
  assert np.shape(data['mean']) == (Nt,Nx,Dx)
  assert np.shape(data['covariance']) == (Nt,Nx,Dx)
  assert np.shape(F) == (Ns,Ds,Nx,Dx)
  assert np.shape(G) == (Ns,Ds,Nv,Nx,Dx)
  p = state_parser(Nst,Ns,Ds,Nv,Nx,Dx)

  if param['solver'] == 'bvls':
    solver = modest.bvls
    upper_bound = 1e6*np.ones(p['total'])
    lower_bound = -1e6*np.ones(p['total'])

    # all inferred fluidities will be positive
    lower_bound[p['fluidity']] = 0

    # all inferred left lateral slip will be positive
    left_lateral_indices = np.array(p['slip'][:,:,0],copy=True)
    thrust_indices = np.array(p['slip'][:,:,1],copy=True)
    upper_bound[left_lateral_indices] = 0.0
    solver_args = (lower_bound,upper_bound)
    solver_kwargs = {}

  elif param['solver'] == 'lstsq':
    solver = modest.lstsq
    solver_args = ()
    solver_kwargs = {'overwrite_a':True,'overwrite_b':True}


  #fprior = prior.create_formatted_prior(param,p,slip_model='parameterized')
  Xprior,Cprior = prior.create_prior(param,p,slip_model='parameterized')   
  reg_matrix = reg.create_regularization(param,p,slip_model='parameterized')
  reg_rows = len(reg_matrix)  
   
  # copy over data file 
  outfile.create_dataset('data/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('data/mask',shape=np.shape(data['mask']),dtype=bool)
  outfile.create_dataset('data/covariance',shape=np.shape(data['covariance']))
  outfile['data/name'] = data['name'][...]
  outfile['data/position'] = data['position'][...]
  outfile['data/time'] = data['time'][...]

  outfile.create_dataset('predicted/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('predicted/covariance',data=1e-10*np.ones((Nt,Nx,Dx)))
  outfile['predicted/name'] = data['name'][...]
  outfile['predicted/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['predicted/position'] = data['position'][...]
  outfile['predicted/time'] = data['time'][...]

  outfile.create_dataset('tectonic/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('tectonic/covariance',data=1e-10*np.ones((Nt,Nx,Dx)))
  outfile['tectonic/name'] = data['name'][...]
  outfile['tectonic/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['tectonic/position'] = data['position'][...]
  outfile['tectonic/time'] = data['time'][...]

  outfile.create_dataset('elastic/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('elastic/covariance',data=1e-10*np.ones((Nt,Nx,Dx)))
  outfile['elastic/name'] = data['name'][...]
  outfile['elastic/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['elastic/position'] = data['position'][...]
  outfile['elastic/time'] = data['time'][...]

  outfile.create_dataset('viscous/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('viscous/covariance',data=1e-10*np.ones((Nt,Nx,Dx)))
  outfile['viscous/name'] = data['name'][...]
  outfile['viscous/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['viscous/position'] = data['position'][...]
  outfile['viscous/time'] = data['time'][...]

  # Initiate arrays where solution will be stored
  outfile.create_dataset('state/all',shape=(Nt,p['total']))
  outfile.create_dataset('state/baseline_displacement',
                         shape = (Nt,)+np.shape(p['baseline_displacement']))    
  outfile.create_dataset('state/secular_velocity',
                         shape = (Nt,)+np.shape(p['secular_velocity']))    
  outfile.create_dataset('state/slip',
                         shape = (Nt,)+np.shape(p['slip'])[1:])    
  outfile.create_dataset('state/slip_derivative',
                         shape = (Nt,)+np.shape(p['slip'])[1:])    
  outfile.create_dataset('state/fluidity',
                         shape = (Nt,)+np.shape(p['fluidity']))    
  outfile['state/time'] = data['time'][...]

  Xprior,Cprior = modest.nonlin_lstsq(
                      regularization,
                      np.zeros(reg_rows),
                      Xprior, 
                      data_covariance=np.eye(reg_rows),
                      prior_covariance=Cprior,
                      system_args=(reg_matrix,),
                      jacobian=regularization_jacobian,
                      jacobian_args=(reg_matrix,),
                      maxitr=param['maxitr'],
                      solver=solver, 
                      solver_args=solver_args,
                      solver_kwargs=solver_kwargs,
                      LM_damping=True,
                      output=['solution','solution_covariance'])

  time_indices = range(Nt)
  block_time_indices = modest.misc.divide_list(time_indices,param['time_blocks'])
  for i in block_time_indices:
    outfile['data/mean'][i,...] = data['mean'][i,...]
    outfile['data/mask'][i,...] = data['mask'][i,...]
    outfile['data/covariance'][i,...] = data['covariance'][i,...]
    di = data['mean'][i,...]
    di = flat_data(di)

    di_mask = np.array(data['mask'][i,:],dtype=bool)
    # expand to three dimensions
    di_mask = np.repeat(di_mask[...,None],3,-1)
    di_mask = flat_data(di_mask) 

    Cdi = data['covariance'][i,...]
    Cdi = flat_data(Cdi)
    data_indices = np.nonzero(~di_mask)[0]

    Xprior,Cprior = modest.nonlin_lstsq(
                      observation,
                      di,
                      Xprior, 
                      data_covariance=Cdi,
                      prior_covariance=Cprior,
                      system_args=(time_shifted[i],F,G,p,slip_func),
                      jacobian=observation_jacobian,
                      jacobian_args=(time_shifted[i],F,G,p,slip_func,slip_jac),
                      data_indices=data_indices,
                      solver=solver, 
                      solver_args=solver_args,
                      solver_kwargs=solver_kwargs,
                      maxitr=param['maxitr'],
                      LM_damping=True,
                      LM_param=1.0,
                      rtol=1e-2,
                      atol=1e-2,
                      output=['solution','solution_covariance'])

  post_mean,post_cov = Xprior,Cprior
  for i in range(Nt):
    outfile['state/all'][i,:] = post_mean
    outfile['state/baseline_displacement'][i,...] = post_mean[p['baseline_displacement']]   
    outfile['state/secular_velocity'][i,...] = post_mean[p['secular_velocity']]   
    outfile['state/slip'][i,...] = slip_func(post_mean[p['slip']],time_shifted[i])   
    outfile['state/slip_derivative'][i,...] = slip_func(post_mean[p['slip']],time_shifted[i],diff=1)   
    outfile['state/fluidity'][i,...] = post_mean[p['fluidity']]   

  # compute predicted data
  logger.info('computing predicted data')
  error = 0.0
  count = 0
  for i in range(Nt):
    predicted = observation_t(post_mean,
                              time_shifted[i],
                              F,G,p,slip_func,flatten=False)
    residual = outfile['data/mean'][i,...] - predicted
    covariance = outfile['data/covariance'][i,...]
    data_mask = np.array(outfile['data/mask'][i,...],dtype=bool)
    error += L2(residual,covariance,data_mask) 
    count += len(np.nonzero(~data_mask)[0])

    outfile['predicted/mean'][i,...] = predicted

    mask = np.zeros(p['total'])
    mask[p['secular_velocity']] = 1.0
    mask[p['baseline_displacement']] = 1.0
    mask_post_mean = post_mean*mask
    outfile['tectonic/mean'][i,...] = observation_t(
                                        mask_post_mean,
                                        time_shifted[i],
                                        F,G,p,
                                        slip_func,
                                        flatten=False)
    mask = np.zeros(p['total'])
    mask[p['slip']] = 1.0
    mask_post_mean = post_mean*mask
    outfile['elastic/mean'][i,...] = observation_t(
                                       mask_post_mean,
                                       time_shifted[i],
                                       F,G,p,
                                       slip_func,  
                                       flatten=False)
    
    visc = (outfile['predicted/mean'][i,...] - 
            outfile['tectonic/mean'][i,...] - 
            outfile['elastic/mean'][i,...])
    outfile['viscous/mean'][i,...] = visc

  logger.info('total RMSE: %s' % np.sqrt(error/count))

  return 


