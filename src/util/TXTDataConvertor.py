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
		self.SetTxtInteracters(txtPathList)

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
		self.__reOpenTimes = []
		for i in range(self.__batchSize):
			frame, trainingData, label, uniqueID = self.__InitSingleTrainingData(self.__txtInteracter[i])
			self.__lastFrame.append(frame)
			self.__trainingData.append(trainingData)
			self.__pannedData.append(trainingData)
			self.__labels.append(label)
			self.__uniqueID.append(uniqueID)
			self.__trainID.append(uniqueID)
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
		return frame, meas, relativeTruth, uids

	def GetMultipleTrainningData(self):
		for i in range(self.__batchSize):
			try:
				self.__lastFrame[i], self.__trainingData[i], self.__pannedData[i], self.__labels[i], self.__uniqueID[i], self.__trainID[i] = \
					self.__GetSingleTrainningData(self.__txtInteracter[i], self.__lastFrame[i], self.__trainingData[i], self.__uniqueID[i])
			except IndexError:
				self.__txtInteracter[i].ReOpenFile()
				self.__reOpenTimes[i] += 1
				print(f'Source file for Batch {i} Reopened {self.__reOpenTimes[i]}th times.')
				self.__lastFrame[i], self.__trainingData[i], self.__pannedData[i], self.__labels[i], self.__uniqueID[i], self.__trainID[i] = \
					self.__GetSingleTrainningData(self.__txtInteracter[i], self.__lastFrame[i], self.__trainingData[i], self.__uniqueID[i])
		return self.__pannedData, self.__labels, self.__trainID

	def __GetSingleTrainningData(self, txtInteracter: TXTInteracter, lastFrame, measList, uids):
		# Get data of current frame
		frame, uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth = txtInteracter.ReadFrame()
		relativeMeas, relativeTruth, uniqueID = self.__GetRelativeData(uniqueID, sensorPosMeas, sensorPosTruth, targetPosMeas, targetPosTruth)
		# Append the current data and shift the data list temporal
		deleteIdx = np.round(measList[:, 2]) == 0
		measList = np.delete(measList, deleteIdx, 0)
		uids = np.delete(uids, deleteIdx, 0)
		measList[:, 2] = np.round(measList[:, 2]) - 1
		measList = np.r_[measList, np.c_[relativeMeas, np.zeros([relativeMeas.shape[0], 1]) + self.__nTimeStep-1]]
		uids = np.r_[uids, uniqueID]
		# Reflash at the end of a track
		if frame < lastFrame:
			frame, measList, relativeTruth, uids = self.__InitSingleTrainingData(txtInteracter)
		# Frame Sampling
		trainMeas = measList.copy()
		trainID = uids.copy()
		dropIdx = np.mod((self.__nTimeStep - np.round(trainMeas[:, 2]) - 1), self.__frameSampleRate).astype(bool)
		# dropIdx = np.mod(np.arange(trainMeas.shape[0]), self.__frameSampleRate).astype(bool)
		# dropIdx = dropIdx[::-1]
		trainMeas = np.delete(trainMeas, dropIdx, 0)
		trainID = np.delete(trainID, dropIdx, 0)
		trainMeas[:, 2] = np.floor_divide(trainMeas[:, 2], self.__frameSampleRate)
		# Pan the measurements & ground truths spatial according to the current measurements
		panValue = np.min(trainMeas[:, :2], 0)
		trainMeas[:, :2] -= panValue
		relativeTruth -= panValue
		return frame, measList, trainMeas, relativeTruth, uids, trainID

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