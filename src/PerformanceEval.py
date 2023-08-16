from __future__ import annotations
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import torch
import os
import warnings
from util.TXTDataConvertor import TXTInteracter
from util.MT3DataConvertor import MT3DataConvertor
from scipy.optimize import linear_sum_assignment

# TEST
# txtInteracter = TXTInteracter('')

class TrackManagement:
	def __init__(self, deadPeriod, maxObject, dimPred=3) -> None:
		self.__deadPeriod = deadPeriod
		self.__dimPred = dimPred
		self.trackArray = np.zeros([maxObject, 2*dimPred+1])
		self.trackArray = np.c_[np.arange(maxObject), self.trackArray]
	
	def PushPred(self, inputPred):
		if (inputPred is None) or (inputPred.size == 0):
			predIdx = []
		else:
			inputPred = inputPred[np.argsort(inputPred[:, 0])]				# sort via ID
			predIdx = np.round(inputPred[:, 0]).astype(int)					# tracks which have measurement

			# Velocity: 0 if flag==0; else delta position / time
			self.trackArray[predIdx, 1+self.__dimPred:1+2*self.__dimPred] = self.trackArray[predIdx, -1, None] \
																		  * (inputPred[:, 1:] - self.trackArray[predIdx, 1:1+self.__dimPred]) \
																		  / (self.__deadPeriod - self.trackArray[predIdx, -1, None] + 1)
			self.trackArray[predIdx, 1:1+self.__dimPred] = inputPred[:, 1:]
			self.trackArray[predIdx, -1] = self.__deadPeriod

		trackIdx = np.round(self.trackArray[:, -1]).astype(int) != 0		# All tracked Tracks
		missIdx = trackIdx.copy()
		missIdx[predIdx] = False											# Tracking but not meausured means missing
		self.trackArray[missIdx, 1:1+self.__dimPred] += self.trackArray[missIdx, 1+self.__dimPred:1+2*self.__dimPred]	# position via velocity
		self.trackArray[missIdx, -1] -= 1									# reduce life cycle

		return self.trackArray[trackIdx, :-1].copy()

class TrackPred:

	def __init__(self, model, taskPath: str, modelPath: str, params=None, matPath = 'None'):
		self.__model = model
		self.mt3DataConvertor = MT3DataConvertor(taskPath, modelPath, training=False, params=params, matPath=matPath)

	def __GetInputMeasurement(self):
		_, uniqueID, sensorPosMeas, _, targetPosMeas, _ = None	# txtInteracter.ReadFrame()
		return uniqueID, sensorPosMeas, targetPosMeas

	def RelativeStatePred(self, existanceThreshold, externalInput = None):
		if externalInput is not None:
			batch, _, _ = self.mt3DataConvertor.Get_batch(externalInput)
			targetPosMeas = externalInput[1]
		else:
			sensorPosMeas, targetPosMeas = self.__GetInputMeasurement()
			batch, _, _ = self.mt3DataConvertor.Get_batch([sensorPosMeas, targetPosMeas])
		if batch is not None:
			output, _, _, _, _ = self.__model.forward(batch, None)
			output_state = output['state'].detach()
			output_logits = output['logits'].sigmoid().detach()
			alive_idx = output_logits[0, :].squeeze(-1) > existanceThreshold
			alive_output = output_state[0, alive_idx, :].cpu().numpy()
			return alive_output
		return targetPosMeas
	
	def PredWithMeasAsso(self, externalInput = None, absTargetPos = False):
		if externalInput is not None:
			uid, sensorPosMeas, targetPosMeas = externalInput
		else:
			uid, sensorPosMeas, targetPosMeas = self.__GetInputMeasurement()

		batch, panValue, _, _ = self.mt3DataConvertor.Get_batch([sensorPosMeas, targetPosMeas])

		if batch is None:
			outputState = np.c_[uid, self.__Sph2Cart(targetPosMeas), np.zeros([targetPosMeas.shape[0], 1]) + 2]
		else:
			# Predict
			output, _, _, _, _ = self.__model.forward(batch, panValue, None)
			output_state = output['state'].detach().cpu() + torch.Tensor(panValue)
			output_logits = output['logits'].detach().cpu().sigmoid().flatten(0,1)

			# Associate results
			meas = [torch.Tensor(self.__Sph2Cart(targetPosMeas))]
			permutation_idx, cost = self.compute_hungarian_matching(output_state, output_logits, meas)
			outIdx, tgtIdx = permutation_idx[0]
			predIdx = outIdx[torch.argsort(tgtIdx)]
			outputState = output_state.squeeze()[predIdx].numpy()

			if absTargetPos:
				outputState += sensorPosMeas[:-1]
			
			outputState = np.c_[uid, outputState, np.zeros([outputState.shape[0], 1]) + 2]

		return outputState
	
	def PredForEval(self, epoch, matFile=None):
		stateList = []
		labelList = []
		for i in range(epoch):
			# Gen meas data
			batch, panValue, labels, unique_ids = self.mt3DataConvertor.Get_batch()

			if batch is not None:
				# Predict
				output, _, _, _, _ = self.__model.forward(batch, panValue, None)
				output_state = output['state'].detach().cpu() + torch.Tensor(panValue)
				output_logits = output['logits'].detach().cpu().sigmoid().flatten(0,1)

				# Gen Ground Truth
				labels[0] = labels[0].cpu()
				outputLabel = labels[0].numpy()

				# Associate results
				permutation_idx, cost = self.compute_hungarian_matching(output_state, None, labels)
				outIdx, tgtIdx = permutation_idx[0]
				predIdx = outIdx[torch.argsort(tgtIdx)]
				outputState = output_state.squeeze()[predIdx].numpy()
			else:
				outputState = outputLabel = np.nan
			
			stateList.append(outputState)
			labelList.append(outputLabel)

		if matFile is not None:
			try:
				scio.savemat(matFile, {'Predict': stateList, 'Truth': labelList})
			except Exception:
				warnings.warn('Error occured, CANNOT save MAT file, please save the returned data manually.')
		
		return stateList, labelList
	
	def compute_hungarian_matching(self, output_state, output_logits, targets):
		""" Performs the matching

		Params:
			outputs: dictionary with 'state' and 'logits'
				state: Tensor of dim [batch_size, num_queries, d_label]
				logits: Tensor of dim [batch_size, num_queries, number_of_classes]

			targets: This is a list of targets (len(targets) = batch_size), where each target is a
					tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
					objects in the target)

		Returns:
			A list of size batch_size, containing tuples of (index_i, index_j) where:
				- index_i is the indices of the selected predictions (in order)
				- index_j is the indices of the corresponding selected targets (in order)
			For each batch element, it holds:
				len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
		"""

		bs, num_queries = output_state.shape[:2]

		# We flatten to compute the cost matrices in a batch
		# [batch_size * num_queries, d_label]
		out = output_state.flatten(0, 1)

		# Also concat the target labels
		# [sum(num_objects), d_labels]
		tgt = torch.cat(targets)

		# Compute the L2 cost
		# [batch_size * num_queries, sum(num_objects)]
		cost = torch.pow(input=torch.cdist(out, tgt, p=2), exponent=1)
		# cost -= output_logits
		if output_logits is not None:
			cost *= (1 - output_logits)

		# Reshape
		# [batch_size, num_queries, sum(num_objects)]
		#cost = cost.view(bs, num_queries, -1)
		cost = cost.view(bs, num_queries, -1).cpu()

		# List with num_objects for each training-example
		sizes = [len(v) for v in targets]

		# Perform hungarian matching using scipy linear_sum_assignment
		with torch.no_grad():
			indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
			permutation_idx = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

		return permutation_idx, cost
	
	def __Sph2Cart(self, targetPosMeas):
		targetThetaRad = targetPosMeas[:, [1]] * np.pi / 180.0
		targetPhiRad = targetPosMeas[:, [2]] * np.pi / 180.0
		targetXMeas = -targetPosMeas[:, [0]] * np.cos(targetPhiRad) * np.cos(targetThetaRad)
		targetYMeas = -targetPosMeas[:, [0]] * np.cos(targetPhiRad) * np.sin(targetThetaRad)
		return np.c_[targetXMeas, targetYMeas]
