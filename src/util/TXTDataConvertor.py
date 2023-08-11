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
	def __init__(self, txtPathList:list[str], n_timestep:int, batchSize:int, frameSampleRate:int, training = True) -> None:
		self.__nTimeStep = round(n_timestep)
		self.__batchSize = round(batchSize)
		self.__frameSampleRate = round(frameSampleRate)
		self.__training = training
		if self.__training:
			self.SetTxtInteracters(txtPathList)
		else:
			assert self.__batchSize == 1, f'Batch Size for Predict should be 1, but got {self.__batchSize}.'
			self.__trainingData = np.zeros([0, 3])
			self.__initFlag = self.__nTimeStep

	def SetTxtInteracters(self, txtPathList:list[str]):
		self.__txtInteracter = [TXTInteracter(txtPathList[i], self.__training) for i in range(self.__batchSize)]
		self.InitMultipleTrainingData()

	def InitMultipleTrainingData(self):
		self.__lastFrame = []
		self.__trainingData = []
		self.__pannedData = []
		self.__labels = []
		self.__uniqueID = []
		self.__trainID = []
		self.__panValue = []
		self.__reOpenTimes = []
		self.__initFlag = []
		for i in range(self.__batchSize):
			self.__lastFrame.append(0)
			self.__trainingData.append(np.zeros([0, 3]))
			self.__pannedData.append(0)
			self.__labels.append(0)
			self.__uniqueID.append(np.array([]))
			self.__trainID.append(0)
			self.__panValue.append(0)
			self.__reOpenTimes.append(0)
			self.__initFlag.append(self.__nTimeStep)

	def GetMultipleTrainningData(self):
		for i in range(self.__batchSize):
			try:
				self.__lastFrame[i], self.__initFlag[i], self.__trainingData[i], self.__pannedData[i], self.__panValue[i], self.__labels[i], self.__uniqueID[i], self.__trainID[i]  = \
					self.__GetSingleTrainningData(self.__txtInteracter[i], self.__lastFrame[i], self.__initFlag[i], self.__trainingData[i], self.__uniqueID[i])
			except IndexError:
				self.__txtInteracter[i].ReOpenFile()
				self.__reOpenTimes[i] += 1
				print(f'Source file for Batch {i} Reopened {self.__reOpenTimes[i]}th times.')
				self.__lastFrame[i], self.__initFlag[i], self.__trainingData[i], self.__pannedData[i], self.__panValue[i], self.__labels[i], self.__uniqueID[i], self.__trainID[i] = \
					self.__GetSingleTrainningData(self.__txtInteracter[i], self.__lastFrame[i], self.__initFlag[i], self.__trainingData[i], self.__uniqueID[i])
		return self.__pannedData, self.__panValue, self.__labels, self.__trainID
	
	def GetMultiplePredictData(self, externalInput:tuple):
		'''
		param:
			externalInput: tuple[sensorPosMeas, targetPosMeas]
		'''

		_, self.__trainingData, self.__pannedData, panValue, _, _ = \
			self.__GetSingleTrainningData(None, None, self.__trainingData, None, externalInput)
		
		if self.__initFlag == 0:
			if np.all(np.round(self.__trainingData[:, 2]) == 0):
				self.__initFlag = self.__nTimeStep
			else:
				return self.__pannedData, panValue
		
		return None, None

	def __GetSingleTrainningData(self, txtInteracter: TXTInteracter, lastFrame, initFlag, measList, uids, externalInput:tuple=None):
		# Get data of current frame
		if self.__training:
			frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = txtInteracter.ReadFrame()
			relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
		else:
			relativeTruth = trainID = None
			frame = lastFrame
			sensorPosMeas, targetPosMeas = externalInput
			if targetPosMeas is not None:
				relativeMeas, _, _ = self.__GetRelativeData(None, sensorPosMeas, None, targetPosMeas, None)
			else:
				relativeMeas = None
			# end if
		# end if

		# Reflash at the end of a track
		if self.__training and frame < lastFrame:
			initFlag = self.__nTimeStep
			measList = np.zeros([0, 3])
			uids = np.array([])

		if initFlag == 0:
			# Append the current data and shift the data list temporal
			deleteIdx = np.round(measList[:, 2]) == 0
			measList = np.delete(measList, deleteIdx, 0)
			measList[:, 2] = np.round(measList[:, 2]) - 1

			if relativeMeas is not None:
				measList = np.r_[measList, np.c_[relativeMeas, np.zeros([relativeMeas.shape[0], 1]) + self.__nTimeStep - 1]]

			if self.__training:
				uids = np.delete(uids, deleteIdx, 0)
				uids = np.r_[uids, uniqueID]
		else:
			measList = np.r_[measList, np.c_[relativeMeas, np.zeros([relativeMeas.shape[0], 1]) + self.__nTimeStep - initFlag]]
			if self.__training:
				uids = np.r_[uids, uniqueID]

		# Frame Sampling
		trainMeas = measList.copy()
		dropBase = int(round(max(trainMeas[:, 2]))) if initFlag != 0 else self.__nTimeStep - 1
		dropIdx = np.mod((dropBase - np.round(trainMeas[:, 2])), self.__frameSampleRate).astype(bool)
		trainMeas = np.delete(trainMeas, dropIdx, 0)
		trainMeas[:, 2] = np.floor_divide(trainMeas[:, 2], self.__frameSampleRate)
		if self.__training:
			trainID = uids.copy()
			trainID = np.delete(trainID, dropIdx, 0)

		# Pan the measurements & ground truths spatial according to the current measurements
		if trainMeas.shape[0] != 0:
			panValue = np.min(trainMeas[:, :2], 0)
			trainMeas[:, :2] -= panValue
		else:
			panValue = np.zeros([1, 2])

		if initFlag != 0:
			if int(round(max(trainMeas[:, 2]))) < 2:
				trainMeas = trainID = None
			initFlag -= 1

		if self.__training:
			relativeTruth -= panValue
			return frame, initFlag, measList, trainMeas, panValue, relativeTruth, uids, trainID
		else:
			return frame, measList, trainMeas, panValue, uids, trainID

	def plotTrain(self):
		frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = self.__txtInteracter[0].ReadFrame()
		relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
		plt.scatter(relativeMeas[:, 0], relativeMeas[:, 1], marker='x')
		plt.scatter(relativeTruth[:, 0], relativeTruth[:, 1], marker='+')
		
	def __GetRelativeData(self, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth):
		targetThetaRad = targetPosMeas[:, [1]] * np.pi / 180.0
		targetPhiRad = targetPosMeas[:, [2]] * np.pi / 180.0
		targetXMeas = -targetPosMeas[:, [0]] * np.cos(targetPhiRad) * np.cos(targetThetaRad)
		targetYMeas = -targetPosMeas[:, [0]] * np.cos(targetPhiRad) * np.sin(targetThetaRad)
		# relativeMeas = np.c_[targetXMeas, targetYMeas] + sensorPosMeas[:2]
		relativeMeas = np.c_[targetXMeas, targetYMeas]
		if self.__training:
			# relativeTruth = targetPosTruth[:, :2]
			relativeTruth =  targetPosTruth[:, :2] - sensorPosTruth[:2]
			relativeTruth = np.delete(relativeTruth, uniqueID == -1, 0)
		else:
			relativeTruth = None
		return relativeMeas, relativeTruth, uniqueID