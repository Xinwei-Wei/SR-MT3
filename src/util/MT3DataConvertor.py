from __future__ import annotations
from util.TXTDataConvertor import TXTDataConvertor
import numpy as np
import torch
from torch import Tensor
from util.misc import NestedTensor

class MT3DataConvertor():
	def __init__(self, txtPathList:list[str], n_timestep:int, batchSize:int, frameSampleRate:int, device = 'cuda', training = True) -> None:
		self.__nTimeStep = round(n_timestep)
		self.__batchSize = round(batchSize)
		self.__frameSampleRate = round(frameSampleRate)
		self.__training = training
		self.device = device
		self.__txtDataConvertor = TXTDataConvertor(txtPathList, self.__nTimeStep, self.__batchSize, self.__frameSampleRate, self.__training)
		
	def Get_batch(self):
		training_data, labels, unique_ids = self.__txtDataConvertor.GetMultipleTrainningData()

		labels = [Tensor(l).to(torch.device(self.device)) for l in labels]
		unique_ids = [list(u) for u in unique_ids]

		# Pad training data
		max_len = max(list(map(len, training_data)))
		training_data, mask = self.__pad_to_batch_max(training_data, max_len)

		# Pad unique ids
		for i in range(len(unique_ids)):
			unique_id = unique_ids[i]
			n_items_to_add = max_len - len(unique_id)
			unique_ids[i] = np.concatenate([unique_id, [-2] * n_items_to_add])[None, :]
		unique_ids = np.concatenate(unique_ids)

		training_nested_tensor = NestedTensor(Tensor(training_data).to(torch.device(self.device)),
											  Tensor(mask).bool().to(torch.device(self.device)))
		unique_ids = Tensor(unique_ids).to(self.device)

		# 防止全0导致的NaN
		for i in range(self.__batchSize):
			if torch.all(training_nested_tensor.tensors[i, :, 2] == 0):
				self.__txtDataConvertor.InitMultipleTrainingData()
				return self.Get_batch()

		return training_nested_tensor, labels, unique_ids
	
	def __pad_to_batch_max(self, training_data, max_len):
		batch_size = len(training_data)
		d_meas = training_data[0].shape[1]
		training_data_padded = np.zeros((batch_size, max_len, d_meas))
		mask = np.ones((batch_size, max_len))
		for i, ex in enumerate(training_data):
			training_data_padded[i,:len(ex),:] = ex
			mask[i,:len(ex)] = 0

		return training_data_padded, mask
	
	def __Polar2Rect(self, radii, angles):
		rect = radii * np.exp(1j * angles)
		return np.real(rect), np.imag(rect)