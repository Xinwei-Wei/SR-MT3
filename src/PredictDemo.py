import os
import argparse
import time
import numpy as np
import torch
from util.misc import NestedTensor, super_load
from util.SetRandomSeed import SetupSeed
from PerformanceEval import TrackPred, TrackManagement

os.environ['CUDA_VISIBLE_DEVICES']='0'
SetupSeed(3407)

parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', default='')
args = parser.parse_known_args()[0]
print(f'Evaluating results from folder: {args.result_filepath}...')

model, params = super_load(args.result_filepath, verbose=True)
trackPred = TrackPred(model, None, None, params)

pred, truth = trackPred.PredForEval(epoch=20000, matFile=None)