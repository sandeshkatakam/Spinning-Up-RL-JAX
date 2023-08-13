import spinningup
from spinningup.user_config import DEFAULT_BACKEND
from spinningup.utils.run_utils import ExperimentalGrid
from spinningup.utils.serialization_utils import convert_json
import argparse
import gym
import json
import os, subprocess, sys
import os.path as osp
import string
import tensorflow as tf
import jax
from copy import deepcopy
from textwrap import dedent


RUN_KEYS = ['num_cpu', 'data_dir', 'datestamp']

SUBSTITUTIONS = {'env':'env_name',
                 'hid': 'ac_kwargs:hidden_sizes',
                 'act': 'ac_kwargs:activation',
                 'cpu': 'num_cpu',
                 'dt' : 'datestamp'}

MPI_COMPATIBLE_ALGOS = ['vpg', 'trpo', 'ppo'}

BASE_ALGO_NAMES = ['vpg', 'trpo', 'ppo', 'ddpg', 'td3', 'sac'}

def add_with_backends(algo_list):
    #helper functions to build lists with backend-specific function names
def friendly_error(err_msg):
    # This adds white space to error message to make it more readable
    return '\n\n' + err_msg + '\n\n'

def parse_and_execute_grid_search(cmd, args):
    """Interprets algorithm name and cmd line args into an ExperimentalGrid"""
    
    if cmd in BASE_ALGO_NAMES:
        backend = DEFAULT_BACKEND[cmd]
        print('\n\n Using default backend (%s) for %s. \n'%(backend, cmd))
        cmd = cmd + '_' + backend
    algo = eval('spinup.'+cmd)

    # Before all else, check to see if any of the flags is 'help'
    valid_help = ['--help', '-h', 'help']
    if any([arg in valid_help for arg in args]):
        print('\n\n Showing docstring for spinup.'+ cmd + ':\n')
        print(algo.__doc__)
        sys.exit()

    def preprocess(arg):
        # Process  an arg by eval-ing it, so users can specify more than just strings at the command line (eg allows for users to give functions as args).
        try:
            return eval(arg)
        except:
            return arg

    arg_dict = dict()
    for i, arg in enumerate(args):
        assert i>0 or '--' in arg, \
                friendly_err("You didn't specify a first flag")
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(process(arg))

