#!/usr/bin/env python
import sys
import argparse
import psgi.stochastic
import psgi.parameterized
import h5py
import json
import logging
import os
import modest.timing as timing

timing.tic('PSGI')
# Setup command line argument parser
p = argparse.ArgumentParser(
      description='PostSeismic Geodetic Inversion')

p.add_argument('-v','--verbose',action='count',
               help='''controls verbosity''')
p.add_argument('--slip_model',type=str,default='stochastic')
p.add_argument('--data_file',type=str,default='data.h5')
p.add_argument('--gf_file',type=str,default='greens_functions.h5')
p.add_argument('--out_file',type=str,default='out.h5')
p.add_argument('--maxitr',type=int,default=10)
p.add_argument('--coseismic_times',nargs='+',type=float)
p.add_argument('--afterslip_start_times',nargs='+',type=float)
p.add_argument('--afterslip_end_times',nargs='+',type=float)
p.add_argument('--time_blocks',type=int,default=1)
p.add_argument('--secular_velocity_mean',type=float,default=0.0)
p.add_argument('--secular_velocity_variance',type=float,default=1e3)
p.add_argument('--baseline_displacement_mean',type=float,default=0.0)
p.add_argument('--baseline_displacement_variance',type=float,default=1e3)
p.add_argument('--initial_slip_mean',type=float,default=0.0)
p.add_argument('--initial_slip_variance',type=float,default=1e3)
p.add_argument('--fluidity_mean',type=float,default=0.0)
p.add_argument('--fluidity_variance',type=float,default=1e3)
p.add_argument('--slip_regularization',type=float,default=1.0)
p.add_argument('--fluidity_regularization',type=float,default=1.0)
p.add_argument('--slip_regularization_order',type=int,default=0)
p.add_argument('--fluidity_regularization_order',type=int,default=0)
p.add_argument('--solver',type=str,default='lstsq')


if os.path.exists('config.json'):
  config_file = open('config.json','r')
  config_default = json.load(config_file)
else:
  config_default = {}

p.set_defaults(**config_default)
config = vars(p.parse_args())


# Setup logger
logger = logging.getLogger()
formatter = logging.Formatter(
              '%(asctime)s %(module)s: [%(levelname)s] %(message)s',
              '%m/%d/%Y %H:%M:%S')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler('log','w')
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

if config['verbose'] >= 1:
  logger.setLevel(logging.DEBUG)
else:
  logger.setLevel(logging.INFO)

string = ''.join(['\n    %-25s %s' % (key+':',val) for key,val in config.iteritems()])
logger.info('arguments: %s' % string)

# load input files
data_file = h5py.File(config['data_file'],'r')

gf_file = h5py.File(config['gf_file'],'r')

out_file = h5py.File(config['out_file'],'w')


if config['slip_model'] == 'stochastic':
  out = psgi.stochastic.main(data_file,
                             gf_file,
                             config,
                             out_file)

out_file = h5py.File(config['output'],'w')


elif config['slip_model'] == 'parameterized':
  out = psgi.parameterized.main(data_file,
                             gf_file,
                             config,
                             out_file)

data_file.close()
gf_file.close()
out_file.close()

logger.info('output saved to out.h5')
timing.summary()



  
