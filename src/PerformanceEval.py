from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from util.MT3DataConvertor import MT3DataConvertor
from scipy.optimize import linear_sum_assignment

# TEST
from util.TXTDataConvertor import TXTInteracter
txtInteracter = TXTInteracter(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}' + '/source/Single_situation0.txt')

class TrackPred:

	def __init__(self, model, timeStep, device = 'cuda'):
		self.__model = model
		self.__timeStep = timeStep
		self.__device = device
		self.mt3DataConvertor = MT3DataConvertor(txtPathList=None, n_timestep=self.__timeStep, batchSize=1, device=self.__device, training=False)

	def __GetInputMeasurement(self):
		_, _, sensorPosMeas, _, targetPosMeas, _ = txtInteracter.ReadFrame()	# TEST
		return sensorPosMeas, targetPosMeas

	def RelativeStatePred(self, existanceThreshold):
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
	
	def compute_hungarian_matching(self, outputs, targets):
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

		output_state = outputs['state']
		output_logits = outputs['logits'].sigmoid().flatten(0,1)

		bs, num_queries = output_state.shape[:2]

		# We flatten to compute the cost matrices in a batch
		# [batch_size * num_queries, d_label]
		out = output_state.flatten(0, 1)

		# Also concat the target labels
		# [sum(num_objects), d_labels]
		tgt = torch.cat(targets)

		# Compute the L2 cost
		# [batch_size * num_queries, sum(num_objects)]
		cost = torch.pow(input=torch.cdist(out, tgt, p=2), exponent=self.order)
		cost -= output_logits

		# Reshape
		# [batch_size, num_queries, sum(num_objects)]
		#cost = cost.view(bs, num_queries, -1)
		cost = cost.view(bs, num_queries, -1).cpu()

		# List with num_objects for each training-example
		sizes = [len(v) for v in targets]

		# Perform hungarian matching using scipy linear_sum_assignment
		with torch.no_grad():
			indices = [linear_sum_assignment(
				c[i]) for i, c in enumerate(cost.split(sizes, -1))]
			permutation_idx = [(torch.as_tensor(i, dtype=torch.int64).to(torch.device(self.device)), torch.as_tensor(
				j, dtype=torch.int64).to(self.device)) for i, j in indices]

		return permutation_idx, cost.to(self.device)
