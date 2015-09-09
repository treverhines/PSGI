#!/usr/bin/env python
import numpy as np
import modest
import logging
import h5py
import scipy
from modest import funtime

logger = logging.getLogger(__name__)

def state_parser(Ns,Ds,Nv,Nx,Dx):
  out = {}

  out['Ns'] = Ns
  out['Ds'] = Ds
  out['Nv'] = Nv
  out['Nx'] = Nx
  out['Dx'] = Dx
  out['total'] = 3*Ns*Ds + Nv + 2*Nx*Dx

  slip_integral = range(0,Ns*Ds)
  out['slip_integral'] = np.reshape(slip_integral,(Ns,Ds))

  slip = range(Ns*Ds,2*Ns*Ds)
  out['slip'] = np.reshape(slip,(Ns,Ds))

  slip_derivative = range(2*Ns*Ds,3*Ns*Ds)
  out['slip_derivative'] = np.reshape(slip_derivative,(Ns,Ds))

  out['fluidity'] = range(3*Ns*Ds,3*Ns*Ds + Nv)

  secular_velocity = range(3*Ns*Ds + Nv,3*Ns*Ds + Nv + Nx*Dx)
  out['secular_velocity'] = np.reshape(secular_velocity,(Nx,Dx))

  baseline = range(3*Ns*Ds + Nv + Nx*Dx,3*Ns*Ds + Nv + 2*Nx*Dx)
  out['baseline_displacement'] = np.reshape(baseline,(Nx,Dx))

  return out

def flat_data(u):
  Nx,Dx = np.shape(u)
  u_flat = np.reshape(u,(Nx*Dx))   
  return u_flat

def flat_cov(cov):
  Nx,Dx,Dx = np.shape(cov) 
  cov_flat = scipy.linalg.block_diag(*cov)
  return cov_flat


@funtime
def observation(X,t,F,G,p,flatten=True):
  # This version enforces nonnegativity in viscosity
  tect = (X[p['secular_velocity']]*t + X[p['baseline_displacement']])
  slip = np.einsum('ijkl,ij',F,X[p['slip']])
  visc = np.einsum('ijklm,ij,k',G,X[p['slip_integral']],
                                  np.exp(X[p['fluidity']]))

  out = tect + slip + visc

  if flatten:
    out = flat_data(out)

  return out


@funtime
def regularization(reg_matrix,state,p):
  state = np.array(state,copy=True)
  state[p['fluidity']] = np.exp(state[p['fluidity']])
  out = reg_matrix.dot(state)
  return out


@funtime
def observation_augmented(X,t,F,G,p,reg_matrix,flatten=True):
  out = observation(X,t,F,G,p,flatten)
  reg = regularization(reg_matrix,X,p)
  out = np.hstack((out,reg))
  return out


@funtime
def observation_jacobian(X,t,F,G,p):
  # This version enforces nonnegativity in viscosity
  # change F from slip,slip_dir,disp,disp_dir to 
  # disp,disp_dir,slip,slip_dir
  F = np.einsum('ijkl->klij',F)
  # change G from slip,slip_dir,visc,disp,disp_dir to 
  # disp,disp_dir,visc,slip,slip_dir
  G = np.einsum('ijklm->lmkij',G)
  jac = np.zeros((p['Nx'],p['Dx'],p['total']))
  jac_vel = t*np.eye(p['Dx']*p['Nx']).reshape(p['Nx'],p['Dx'],p['Nx'],p['Dx'])
  jac_disp = np.eye(p['Dx']*p['Nx']).reshape(p['Nx'],p['Dx'],p['Nx'],p['Dx'])
  jac[:,:,p['secular_velocity']] = jac_vel
  jac[:,:,p['baseline_displacement']] = jac_disp
  jac[:,:,p['slip']] = F
  jac[:,:,p['slip_integral']] = np.einsum('ijklm,k',G,np.exp(X[p['fluidity']]))
  # exp(fluidity) is being broadcast along the middle axis in G
  jac[:,:,p['fluidity']] = np.einsum('ij...lm,lm,...->ij...',
                                     G,
                                     X[p['slip_integral']],
                                     np.exp(X[p['fluidity']]))

  jac = np.reshape(jac,(p['Nx']*p['Dx'],p['total']))   
  return jac
  

@funtime
def regularization_jacobian(reg_matrix,state,p):
  '''combines slip and fluidity regularization matrix and modifies
     the fluidity regularization matrix for nonnegativity'''
  #flu_reg = fluidity_regularization(state[p['fluidity']])
  # form the full regularization matrix
  reg_matrix = np.array(reg_matrix,copy=True)
  reg_matrix[:,p['fluidity']] *= np.exp(state[p['fluidity']])
  return reg_matrix


@funtime
def observation_jacobian_augmented(X,t,F,G,p,reg_matrix):
  jac = observation_jacobian(X,t,F,G,p)
  reg = regularization_jacobian(reg_matrix,X,p)
  jac = np.vstack((jac,reg))
  return jac


@funtime
def transition(X,dt,p):
  Xout = np.copy(X)
  Xout[p['slip_integral']] = (X[p['slip_integral']] +
                              X[p['slip']]*dt +
                              X[p['slip_derivative']]*0.5*dt**2)

  Xout[p['slip']] = (X[p['slip']] +
                     X[p['slip_derivative']]*dt)

  return Xout


@funtime
def process_covariance(X,dt,alpha,p):
  Q = np.zeros((p['total'],p['total']))
  Q[p['slip_derivative'],p['slip_derivative']] = alpha**2*dt
  Q[p['slip'],p['slip']] = alpha**2*(dt**3)/3.0
  Q[p['slip_integral'],p['slip_integral']] = alpha**2*(dt**5)/20.0
 
  Q[p['slip_integral'],p['slip']] = alpha**2*(dt**4)/8.0
  Q[p['slip'],p['slip_integral']] = alpha**2*(dt**4)/8.0

  Q[p['slip_derivative'],p['slip']] = alpha**2*(dt**2)/2.0
  Q[p['slip'],p['slip_derivative']] = alpha**2*(dt**2)/2.0
 
  Q[p['slip_derivative'],p['slip_integral']] = alpha**2*(dt**3)/6.0
  Q[p['slip_integral'],p['slip_derivative']] = alpha**2*(dt**3)/6.0

  return Q


def RMSE(residual,covariance,mask):
  if np.all(mask):
    return 0.0

  residual = flat_data(residual[~mask,:])
  covariance = flat_cov(covariance[~mask,:,:])
  #residual = residual[~mask] 
  #covariance = covariance[np.ix_(~mask,~mask)] 
  covinv = np.linalg.inv(covariance)
  #print(covinv)
  return np.sqrt(np.einsum('i,ij,j',residual,covinv,residual))
  #return np.sqrt(np.einsum('i,i',residual,residual))


def form_regularization(reg,p):
  reg_matrix = np.zeros((0,p['total']))
  for i,v in reg.iteritems():
    #if len(np.shape(p[i])) == 2:
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


def form_prior(prior,p):
  Xprior = np.zeros(p['total'])  
  Cprior = 1e-10*np.eye(p['total'],p['total'])
  for i,v in prior.iteritems():
    if i == 'fluidity':
      flu_prior = v['mean'][...]
      flu_prior_var = v['variance'][...] 
      # user specified fluidity variance needs to be converted to 
      # the variance of beta where fluidity = exp(beta) and beta
      # is the fluidity parameter which PSGI estimates.
      
      # \sigma_{beta}^2 = \frac{d log(\fluidity)}{d \fluidty}^2 * \sigma_{fluidity}^2 
      # \sigma_{beta}^2 = \frac{\sigma_{fluidity}^2}{\fluidty}^2}

      # convert the fluidity prior to prior for beta
      beta_prior_var = flu_prior_var/(flu_prior**2)
      beta_prior = np.log(flu_prior)
      vnew = {}
      vnew['mean'] = beta_prior
      vnew['variance'] = beta_prior_var
      v = vnew

    Xprior[p[i]] = v['mean'][...]
    Cprior[p[i],p[i]] = v['variance'][...]

  return Xprior,Cprior

@funtime
def kalmanfilter(data,gf,reg,prior,param,outfile):
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

  alpha = param['slip_acceleration_variance']
  jump_factor = param['jump_variance_factor']
  jump_times = param['jump_times']
  time = data['time'][:]

  Ns,Ds,Nv,Nx,Dx = np.shape(G)
  Nt = len(time)

  # check for consistency between input
  assert np.shape(data['mean']) == (Nt,Nx,Dx)
  assert np.shape(data['covariance']) == (Nt,Nx,Dx,Dx)
  assert np.shape(F) == (Ns,Ds,Nx,Dx)
  assert np.shape(G) == (Ns,Ds,Nv,Nx,Dx)
  p = state_parser(Ns,Ds,Nv,Nx,Dx)

  Xprior,Cprior = form_prior(prior,p)   
  reg_matrix = form_regularization(reg,p)
  reg_rows = len(reg_matrix)  

  kalman = modest.KalmanFilter(
             Xprior,Cprior,
             observation_augmented,
             obs_args=(F,G,p,reg_matrix),
             ojac=observation_jacobian_augmented,
             ojac_args=(F,G,p,reg_matrix),
             trans=transition,
             trans_args=(p,),
             pcov=process_covariance,
             pcov_args=(alpha,p),
             core=False,
             solver_kwargs={'maxitr':param['maxitr']},
             temp_file=param['history_file'])

  
  # prepare outfile file

  # copy over data file 
  outfile.create_dataset('data/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('data/mask',shape=np.shape(data['mask']),dtype=bool)
  outfile.create_dataset('data/covariance',shape=np.shape(data['covariance']))
  outfile['data/name'] = data['name'][...]
  outfile['data/position'] = data['position'][...]
  outfile['data/time'] = data['time'][...]

  outfile.create_dataset('predicted/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('predicted/covariance',shape=np.shape(data['covariance']))
  outfile['predicted/name'] = data['name'][...]
  outfile['predicted/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['predicted/position'] = data['position'][...]
  outfile['predicted/time'] = data['time'][...]

  outfile.create_dataset('tectonic/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('tectonic/covariance',shape=np.shape(data['covariance']))
  outfile['tectonic/name'] = data['name'][...]
  outfile['tectonic/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['tectonic/position'] = data['position'][...]
  outfile['tectonic/time'] = data['time'][...]

  outfile.create_dataset('elastic/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('elastic/covariance',shape=np.shape(data['covariance']))
  outfile['elastic/name'] = data['name'][...]
  outfile['elastic/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['elastic/position'] = data['position'][...]
  outfile['elastic/time'] = data['time'][...]

  outfile.create_dataset('viscous/mean',shape=np.shape(data['mean']))
  outfile.create_dataset('viscous/covariance',shape=np.shape(data['covariance']))
  outfile['viscous/name'] = data['name'][...]
  outfile['viscous/mask'] = np.array(0.0*data['mask'][...],dtype=bool)
  outfile['viscous/position'] = data['position'][...]
  outfile['viscous/time'] = data['time'][...]


  ## THIS TAKES WAAAAAAYYYYYY TOO LONG!!!!!
  # fill in covariance matrices with something small
  
  #for i in range(Nt):
  #  for j in range(Nx):    
  #    outfile['predicted/covariance'][i,j,:,:] = 1e-8*np.eye(3)
  #    outfile['tectonic/covariance'][i,j,:,:] = 1e-8*np.eye(3)
  #    outfile['elastic/covariance'][i,j,:,:] = 1e-8*np.eye(3)
  #    outfile['viscous/covariance'][i,j,:,:] = 1e-8*np.eye(3)
  a = np.ones((Nt,Nx))
  b = np.eye(3)
  c = 1e-10*np.einsum('ij,kl',a,b)
  outfile['predicted/covariance'][...] = c
  outfile['tectonic/covariance'][...] = c
  outfile['elastic/covariance'][...] = c
  outfile['viscous/covariance'][...] = c

  outfile.create_dataset('state/all',shape=(Nt,p['total']))
  for k in ['baseline_displacement',
            'secular_velocity',
            'slip_integral',
            'slip',
            'slip_derivative',
            'fluidity']:
    outfile.create_dataset('state/' + k,shape=(Nt,)+np.shape(p[k]))

  outfile['state/time'] = data['time'][...]

  logger.info('starting Kalman filter iterations')
  for i in range(Nt):
    outfile['data/mean'][i,...] = data['mean'][i,...]
    outfile['data/mask'][i,...] = data['mask'][i,...]
    outfile['data/covariance'][i,...] = data['covariance'][i,...]
    di = flat_data(data['mean'][i,...])
    di = np.hstack((di,np.zeros(reg_rows)))
    di_mask = np.array(data['mask'][i,:],dtype=bool)
    # expand to three dimensions
    di_mask = np.repeat(di_mask[:,None],3,1)
    di_mask = flat_data(di_mask)
    di_mask = np.hstack((di_mask,np.zeros(reg_rows,dtype=bool)))
    Cdi = flat_cov(data['covariance'][i,...])
    Cdi = scipy.linalg.block_diag(Cdi,np.eye(reg_rows))
    if np.all(time[i] < jump_times):
      kalman.pcov_args = (0.0*alpha,p)
      kalman.next(di,Cdi,time[i],mask=di_mask)
      kalman.pcov_args = (alpha,p)

    elif (i != 0) & np.any((time[i] > jump_times) & (time[i-1] <= jump_times)):
      logger.info('increasing slip variance by %sx when updating '
                  'from t=%s to t=%s' % (jump_factor,time[i-1],time[i]))
      kalman.pcov_args = (jump_factor*alpha,p)
      kalman.next(di,Cdi,time[i],mask=di_mask)
      kalman.pcov_args = (alpha,p)

    else:   
      kalman.next(di,Cdi,time[i],mask=di_mask)

  logger.info('starting Rauch-Tung-Striebel smoothing')
  kalman.smooth()
  for i in range(Nt):
    outfile['state/all'][i,:] = kalman.history['smooth'][i,:]
    for k in ['baseline_displacement',
              'secular_velocity',
              'slip_integral',
              'slip',
              'slip_derivative',
              'fluidity']:
      outfile['state/' + k][i,...] = np.array(kalman.history['smooth'])[i,p[k]]

  # compute predicted data
  logger.info('computing predicted data')
  error = 0.0
  for i in range(Nt):
    predicted = observation(outfile['state/all'][i],
                            time[i],
                            F,G,p,flatten=False)
    residual = outfile['data/mean'][i,...] - predicted
    covariance = outfile['data/covariance'][i,...]
    data_mask = np.array(outfile['data/mask'][i,...],dtype=bool)
    error += RMSE(residual,covariance,data_mask) 

    outfile['predicted/mean'][i,...] = predicted

    mask = np.zeros(p['total'])
    mask[p['secular_velocity']] = 1.0
    mask[p['baseline_displacement']] = 1.0
    state = outfile['state/all'][i]*mask
    outfile['tectonic/mean'][i,...] = observation(
                                        state,
                                        time[i],
                                        F,G,p,
                                        flatten=False)

    mask = np.zeros(p['total'])
    mask[p['slip']] = 1.0
    state = outfile['state/all'][i]*mask
    outfile['elastic/mean'][i,...] = observation(
                                       state,
                                       time[i],
                                       F,G,p,
                                       flatten=False)
    
    visc = (outfile['predicted/mean'][i,...] - 
            outfile['tectonic/mean'][i,...] - 
            outfile['elastic/mean'][i,...])
    outfile['viscous/mean'][i,...] = visc

  logger.info('total RMSE: %s' % error)

  return 


