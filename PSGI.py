#!/usr/bin/env python
import sys
import argparse
import psgi.filter
import h5py
import json
import logging
import modest.timing as timing

timing.tic('PSGI')
# Setup command line argument parser
p = argparse.ArgumentParser(
      description='PostSeismic Geodetic Inversion')

p.add_argument('config',type=str,
               help='''name of configuration file''')

p.add_argument('-v','--verbose',action='count',
               help='''controls verbosity''')

args = vars(p.parse_args())
config_name = args['config']

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
if args['verbose'] >= 1:
  logger.setLevel(logging.DEBUG)
else:
  logger.setLevel(logging.INFO)

# load configuration file
f = open(config_name,'r')
config = json.load(f)

# load input files
data_file = h5py.File(config['data'],'r')

gf_file = h5py.File(config['greens_functions'],'r')

reg_file = h5py.File(config['regularization'],'r')

out_file = h5py.File('out.h5','w')

out = psgi.filter.kalmanfilter(data_file,
                               gf_file,
                               reg_file,
                               config,
                               out_file)

data_file.close()
gf_file.close()
reg_file.close()
out_file.close()

logger.info('output saved to out.h5')
timing.summary()



  
