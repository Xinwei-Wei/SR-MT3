from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

class TXTInteracter:
	def __init__(self, txtFilePath:str, training = True):
		self.txtFilePath = txtFilePath
		self.__txtFile = open(self.txtFilePath, 'r')
		self.__training = training

	def ReOpenFile(self, txtFilePath:str = None):
		if txtFilePath is not None:
			self.txtFilePath = txtFilePath
		self.__txtFile.close()
		self.__txtFile = open(self.txtFilePath, 'r')
	
	def CloseFile(self):
		self.__txtFile.close()

	def ReadFrame(self):
		uniqueID = sensorPosTruth = targetPosTruth = None

		# SensorData
		rawData = self.__txtFile.readline().strip().split()
		numData = np.array(rawData).astype(float)
		frame = numData[0].astype(int)
		sensorPosMeas = numData[1:4]
		if self.__training:
			sensorPosTruth = numData[4:]

		# TargetData
		numData = np.zeros([0, 7])
		while True:
			rawData = self.__txtFile.readline().strip().split()
			if len(rawData) == 0:
				break
			numData = np.r_[numData, np.array(rawData).astype(float)[np.newaxis, :]]
		# end while
		targetPosMeas = numData[:, 1:4]
		if self.__training:
			uniqueID = numData[:, 0]
			targetPosTruth = numData[:, 4:]

		return frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth

class TXTDataConvertor:
	def __init__(self, txtPathList:list[str], n_timestep:int, batchSize:int, training = True) -> None:
		self.__nTimeStep = round(n_timestep)
		self.__batchSize = round(batchSize)
		self.__training = training
		self.SetTxtInteracters(txtPathList)

	def SetTxtInteracters(self, txtPathList:list[str]):
		self.__txtInteracter = [TXTInteracter(txtPathList[i], self.__training) for i in range(self.__batchSize)]
		self.InitMultipleTrainingData()

	def InitMultipleTrainingData(self):
		self.__trainingData = []
		self.__labels = []
		self.__uniqueID = []
		self.__reOpenTimes = []
		for i in range(self.__batchSize):
			trainingData, label, uniqueID = self.__InitSingleTrainingData(self.__txtInteracter[i])
			self.__trainingData.append(trainingData)
			self.__labels.append(label)
			self.__uniqueID.append(uniqueID)
			self.__reOpenTimes.append(0)

	def __InitSingleTrainingData(self, txtInteracter: TXTInteracter):
		meas = np.zeros([0, 3])
		uids = np.array([])
		for i in range(self.__nTimeStep):	# TODO Training judgement
			frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = txtInteracter.ReadFrame()
			relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
			meas = np.r_[meas, np.c_[relativeMeas, np.zeros([relativeMeas.shape[0], 1]) + i]]
			uids = np.r_[uids, uniqueID]
		# end for
		return meas, relativeTruth, uids

	def GetMultipleTrainningData(self):
		for i in range(self.__batchSize):	# TODO Reflash at the end of a track
			try:
				self.__trainingData[i], self.__labels[i], self.__uniqueID[i] = \
					self.__GetSingleTrainningData(self.__txtInteracter[i], self.__trainingData[i], self.__uniqueID[i])
			except IndexError:
				self.__txtInteracter[i].ReOpenFile()
				self.__reOpenTimes[i] += 1
				print(f'Source file for Batch {i} Reopened {self.__reOpenTimes[i]}th times.')
				self.__trainingData[i], self.__labels[i], self.__uniqueID[i] = \
					self.__GetSingleTrainningData(self.__txtInteracter[i], self.__trainingData[i], self.__uniqueID[i])
		return self.__trainingData, self.__labels, self.__uniqueID

	def __GetSingleTrainningData(self, txtInteracter: TXTInteracter, meas, uids):
		frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = txtInteracter.ReadFrame()
		relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
		deleteIdx = np.round(meas[:, 2]) == 0
		meas = np.delete(meas, deleteIdx, 0)
		uids = np.delete(uids, deleteIdx, 0)
		meas[:, 2] = np.round(meas[:, 2]) - 1
		meas = np.r_[meas, np.c_[relativeMeas, np.zeros([relativeMeas.shape[0], 1]) + self.__nTimeStep-1]]
		uids = np.r_[uids, uniqueID]
		return meas, relativeTruth, uids

	def plotTrain(self):
		frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = self.__txtInteracter[0].ReadFrame()
		relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
		plt.scatter(relativeMeas[:, 0], relativeMeas[:, 1], marker='x')
		plt.scatter(relativeTruth[:, 0], relativeTruth[:, 1], marker='+')
		
	def __GetRelativeData(self, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth):
		targetThetaRad = targetPosMeas[:, [1]] * np.pi / 180.0
		targetPhiRad = targetPosMeas[:, [2]] * np.pi / 180.0
		# targetXMeas = targetPosMeas[:, [0]] * np.sin(targetThetaRad) * np.cos(targetPhiRad)
		# targetYMeas = targetPosMeas[:, [0]] * np.sin(targetThetaRad) * np.sin(targetPhiRad)
		targetXMeas = -targetPosMeas[:, [0]] * np.cos(targetPhiRad) * np.cos(targetThetaRad)
		targetYMeas = -targetPosMeas[:, [0]] * np.cos(targetPhiRad) * np.sin(targetThetaRad)
		# relativeMeas = np.c_[targetXMeas, targetYMeas] + sensorPosMeas[:2]
		relativeMeas = np.c_[targetXMeas, targetYMeas]
		if self.__training:
			# relativeTruth = targetPosTruth[:, :2]
			relativeTruth =  targetPosTruth[:, :2] - sensorPosTruth[:2]
			relativeTruth = np.delete(relativeTruth, uniqueID == -1, 0)
			# uniqueID = np.delete(uniqueID, uniqueID == -1, 0)
		else:
			relativeTruth = None
		return relativeMeas, relativeTruth, uniqueID