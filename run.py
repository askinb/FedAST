import argparse
import client
import config
import logging
import os
import server
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')
args = parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')

fl_config = config.Config(args.config)
# Initialize server
fl_server = server.Server(fl_config)
fl_server.boot()
fl_server.train()