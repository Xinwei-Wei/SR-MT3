from __future__ import annotations
import numpy as np

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
		for i in range(self.__batchSize):
			trainingData, label, uniqueID = self.__InitSingleTrainingData(self.__txtInteracter[i])
			self.__trainingData.append(trainingData)
			self.__labels.append(label)
			self.__uniqueID.append(uniqueID)

	def __InitSingleTrainingData(self, txtInteracter: TXTInteracter):
		meas = np.zeros([0, 3])
		for i in range(self.__nTimeStep):	# TODO Training judgement
			frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = txtInteracter.ReadFrame()
			relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
			meas = np.r_[meas, np.c_[relativeMeas, np.zeros([relativeMeas.shape[0], 1]) + i]]
		# end for
		return meas, relativeTruth, uniqueID

	def GetMultipleTrainningData(self):
		for i in range(self.__batchSize):	# TODO Reflash at the end of a track
			self.__trainingData[i], self.__labels[i], self.__uniqueID[i] = self.__GetSingleTrainningData(self.__txtInteracter[i], self.__trainingData[i])
		return self.__trainingData, self.__labels, self.__uniqueID

	def __GetSingleTrainningData(self, txtInteracter: TXTInteracter, meas):
		frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = txtInteracter.ReadFrame()
		relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
		meas = np.delete(meas, np.round(meas[:, 2]) == 0, 0)
		meas[:, 2] = np.round(meas[:, 2]) - 1
		meas = np.r_[meas, np.c_[relativeMeas, np.zeros([relativeMeas.shape[0], 1]) + self.__nTimeStep-1]]
		return meas, relativeTruth, uniqueID
		
	def __GetRelativeData(self, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth):
		targetXMeas = targetPosMeas[:, [0]] * np.sin(targetPosMeas[:, [1]]) * np.cos(targetPosMeas[:, [2]])
		targetYMeas = targetPosMeas[:, [0]] * np.sin(targetPosMeas[:, [1]]) * np.sin(targetPosMeas[:, [2]])
		relativeMeas = np.c_[targetXMeas, targetYMeas] - sensorPosMeas[:2]
		if self.__training:
			relativeTruth = targetPosTruth[:, :2] - sensorPosTruth[:2]
			relativeTruth = np.delete(relativeTruth, uniqueID == -1, 0)
			uniqueID = np.delete(uniqueID, uniqueID == -1, 0)
		else:
			relativeTruth = None
		return relativeMeas, relativeTruth, uniqueID