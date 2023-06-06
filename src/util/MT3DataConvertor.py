from util.FusionDataGenerator import FusionDataGenerator
import numpy as np
import torch
import scipy.io as scio
from torch import Tensor
from util.load_config_files import load_yaml_into_dotdict
from util.misc import NestedTensor

class MT3DataConvertor():
	def __init__(self, taskPath: str, modelPath: str, evalBS = -1, matPath = 'None') -> None:
		# 从yaml载入超参数
		self.params = load_yaml_into_dotdict(taskPath)
		self.params.update(load_yaml_into_dotdict(modelPath))
		if evalBS == -1:
			self.__batchSize = int(self.params.totalArg.batchSize)
		else:
			self.__batchSize = evalBS
		self.__nTimeStep = int(self.params.totalArg.n_timesteps)
		self.cycle = int(np.floor((self.params.targetArg.tf - self.params.targetArg.t0) * 1000 / self.params.sensorArg.Rect.T) - self.__nTimeStep + 1)
		self.device = self.params.training.device
		if matPath != 'None':
			self.useDataFromMAT = True
			self.mat = scio.loadmat(matPath)['Sensor']
			self.SetEpoch()
		else:
			self.useDataFromMAT = False
			self.fusionDataGenerator = FusionDataGenerator(taskPath, self.__batchSize)
			self.ResetBias()

	def SetEpoch(self, epoch = 0):
		self.epoch = epoch - 1
		self.ResetBias()

	def ResetBias(self, bias=0):
		self.__get_Seq_training_example = self.__Get_Seq_training_example()
		for i in range(bias):
			next(self.__get_Seq_training_example)
		# print('DataGeneratorParams: TimeStep = %d, Cycle = %d, Bias = %d.' %(self.__nTimeStep, self.cycle, bias))
		
	def Get_batch(self):
		training_data, labels, unique_ids = next(self.__get_Seq_training_example)

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
				self.ResetBias()
				return self.Get_batch()

		return training_nested_tensor, labels, unique_ids
	
	def __Converte_Measurements_from_MAT(self):
		epoch = self.epoch
		sensor = self.mat
		frame = self.__nTimeStep + self.cycle
		sensorType = 'Rect'
		
		assert sensor[0, 0].shape[0] == frame, f'Data from MAT must match the Frame of the Model: {sensor[0, 0].shape[0]} & {frame}.'
		assert epoch < sensor.shape[1], f'Epoch: {epoch} should be less than {sensor.shape[1]}.'
		
		fusionData = dict({'Radar': dict(), 'Pradar': dict(), 'DAS': dict(), 'Rect': dict()})
		fusionData[sensorType]['t'] = sensor[0, epoch].squeeze()												# t
		zM = sensor[1, epoch]																					# Z
		zT = sensor[2, epoch]																					# Z_Truth
		id = sensor[3, epoch]																					# ids
		Z = []
		Z_Truth = []
		for i in range(frame):
			zk = np.r_[np.c_[id[i, 0], np.zeros([1, zM[i, 0].shape[1] - id[i, 0].shape[1]]) - 1], zM[i, 0]]		# ids杂波补-1
			zk = zk[:, np.random.permutation(np.arange(zk.shape[1]))]											# 乱序
			Z.append(zk)
			Z_Truth.append(np.r_[id[i, 0], zT[i, 0]])
		# end for
		fusionData[sensorType]['Z'] = Z
		fusionData[sensorType]['Z_Truth'] = Z_Truth
		fusionData = [fusionData]

		return fusionData
	
	def __Get_Seq_training_example(self):
		while True:
			if self.useDataFromMAT:
				self.epoch += 1
				self.fusionData = self.__Converte_Measurements_from_MAT()
			else:
				self.fusionData = self.fusionDataGenerator.GenMeasurements()
			# end if
			for self.bias in range(self.cycle):
				yield self.__Get_batch_training_example(self.fusionData, self.bias)

	def __Get_batch_training_example(self, result, bias):
		batchSize = self.__batchSize
		nTimestep = self.__nTimeStep
		training_data = []
		labels = []
		unique_ids = []

		for k in range(batchSize):
			# tk = result[k]['Radar']['t'][bias : bias+nTimestep]
			# Zk = result[k]['Radar']['Z'][bias : bias+nTimestep]
			# Z_Truth_k = result[k]['Radar']['Z_Truth'][bias+nTimestep]
			tk = result[k]['Rect']['t'][bias : bias+nTimestep]
			Zk = result[k]['Rect']['Z'][bias : bias+nTimestep]
			Z_Truth_k = result[k]['Rect']['Z_Truth'][bias+nTimestep-1]
			training_data_k = np.empty([0, 3])
			unique_ids_k = np.empty(0).astype(np.int64)
			for n in range(nTimestep):
				Z = Zk[n]
				# ZX, ZY = self.__Polar2Rect(Z[1], Z[2])
				# ZTX, ZTY = self.__Polar2Rect(Z_Truth_k[1], Z_Truth_k[2])
				ZX, ZY = (Z[1], Z[2])
				ZTX, ZTY = (Z_Truth_k[1], Z_Truth_k[2])
				training_data_n = np.c_[ZX[:, np.newaxis], ZY[:, np.newaxis], np.zeros([ZX.shape[0], 1]) + tk[n] - tk[0]]
				training_data_k = np.r_[training_data_k, training_data_n]
				unique_ids_k  = np.append(unique_ids_k, Z[0].astype(np.int64))
			# end for
			training_data.append(training_data_k)
			unique_ids.append(unique_ids_k)
			labels.append(np.c_[ZTX[:, np.newaxis], ZTY[:, np.newaxis]])
		return training_data, labels, unique_ids
	
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