import numpy as np

def create_formatted_prior(params,p,slip_model='stochastic'):
  Ns = p['Ns']
  Ds = p['Ds']
  Nv = p['Nv']
  Nx = p['Nx']
  Dx = p['Dx']
  out = {'secular_velocity':{},
         'baseline_displacement':{},
         'slip':{},
         'fluidity':{}}

  # set secular velocity
  sec_vel = np.zeros(np.shape(p['secular_velocity']))
  if params.has_key('secular_velocity_file'):
    sec_vel_file = np.loadtxt(params['secular_velocity_file'],dtype=str)
    sec_vel[:,[0,1]] = np.array(sec_vel_file[:,[1,2]],dtype=float)

  else:
    sec_vel = np.ones(np.shape(p['secular_velocity']))
    sec_vel *= params['secular_velocity_mean']

  sec_vel_var = np.ones(np.shape(p['secular_velocity']))
  sec_vel_var *= params['secular_velocity_variance']
 
  out['secular_velocity']['mean'] = sec_vel
  out['secular_velocity']['variance'] = sec_vel_var

  # set baseline displacement
  baseline = np.ones(np.shape(p['baseline_displacement']))
  baseline *= params['baseline_displacement_mean'] 

  baseline_var = np.ones(np.shape(p['baseline_displacement']))
  baseline_var *= params['baseline_displacement_variance']

  out['baseline_displacement']['mean'] = baseline
  out['baseline_displacement']['variance'] = baseline_var

  slip = np.ones(np.shape(p['slip']))
  slip_var = np.ones(np.shape(p['slip']))

  slip *= params['initial_slip_mean']
  slip_var *= params['initial_slip_variance']

  out['slip']['mean'] = slip
  out['slip']['variance'] = slip_var

  # set fluidity
  fluidity = np.ones(np.shape(p['fluidity']))
  fluidity *= params['fluidity_mean']

  fluidity_var = np.ones(np.shape(p['fluidity']))
  fluidity_var *= params['fluidity_variance']

  out['fluidity']['mean'] = fluidity
  out['fluidity']['variance'] = fluidity_var

  return out


def create_prior(params,p,slip_model='stochastic'):
  prior = create_formatted_prior(params,p,slip_model=slip_model)

  if slip_model=='stochastic':
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

  else:
    Xprior = np.zeros(p['total'])
    Cprior = 1e-10*np.eye(p['total'],p['total'])
    for i,v in prior.iteritems():
      Xprior[p[i]] = v['mean'][...]
      Cprior[p[i],p[i]] = v['variance'][...]

  return Xprior,Cprior
