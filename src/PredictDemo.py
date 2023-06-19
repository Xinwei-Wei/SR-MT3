import os
import argparse
import time
import numpy as np
import torch
from util.misc import NestedTensor, super_load
from util.TXTDataConvertor import TXTInteracter
from PerformanceEval import TrackPred, TrackManagement
import struct
from util.sharedMemory import *

request_formats = "<" + "dddd" + ("d" + "ddd") * 10		# input structure
response_formats = "<" + "d" + ("ddddddd" * 10)			# output structure
request_sizes = struct.calcsize(request_formats)
response_sizes = struct.calcsize(response_formats)
response_shape = len(response_formats) - 1

parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', default='')
args = parser.parse_known_args()[0]
print(f'Evaluating results from folder: {args.result_filepath}...')

model, params = super_load(args.result_filepath, verbose=True)
trackPred = TrackPred(model, params.data_generation.n_timesteps * params.data_generation.frame_sample_rate, params.data_generation.frame_sample_rate, params.training.device)
trackManagement = TrackManagement(deadPeriod=5, maxObject=10, dimPred=3)

missile_id, alg_id = 4, 2
print("missile:" + str(missile_id) + ",alg:" + str(alg_id))
wiconEx = WICONEx(7100, 0x100, 0x184C, missile_id, alg_id)
wiconEx.Open()

while True:
	time.sleep(0.005)
	if wiconEx.HasRequest():
		# Acquire Input
		requestData = wiconEx.ReceiveRequest(request_sizes, request_formats)
		targetNum = round(requestData[0])
		if targetNum != 0:
			rawInput = np.array(requestData).reshape(-1, 4)[:targetNum+1, :]
			uniqueID = np.round(rawInput[1:, 0]).astype(int)
			sensorPosMeas = rawInput[0, 1:]
			targetPosMeas = rawInput[1:, 1:]
		else:
			uniqueID = sensorPosMeas = targetPosMeas = None

		# Predict States
		outputState = trackPred.PredWithMeasAsso([uniqueID, sensorPosMeas, targetPosMeas], absTargetPos=False)
		outputState = trackManagement.PushPred(outputState)

		# Generate Output & Reply
		if outputState is not None:
			output = outputState.flatten()
			output = np.r_[outputState.shape[0], output, np.zeros([response_shape - output.shape[0] - 1])].tolist()
		else:
			output = np.zeros([response_shape]).tolist()
		# end if
		wiconEx.SendReply(output, response_formats)