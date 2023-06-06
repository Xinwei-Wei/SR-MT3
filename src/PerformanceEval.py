import matplotlib.pyplot as plt
import numpy as np
import torch

def CalculateRMSE(dataGenerator, model, cycle, batchSize, epoch, cutoffDistance, existanceThreshold = 0.9):
	nEpoch = 0
	xRMSE = np.zeros([cycle, batchSize])
	yRMSE = np.zeros([cycle, batchSize])
	while nEpoch < epoch:
		dataGenerator.ResetBias()
		for nCycle in range(cycle):
			batch, labels, unique_ids = dataGenerator.Get_batch()
			# if nCycle == 0 and np.size(dataGenerator.fusionData[0]['Radar']['Z_Truth'][50]) == 0:
			if nCycle == 0 and np.size(dataGenerator.fusionData[0]['Rect']['Z_Truth'][-1]) == 0:
				nEpoch -= 1
				break
			output, _, _, _, _ = model.forward(batch, unique_ids)
			output_state = output['state'].detach()
			output_logits = output['logits'].sigmoid().detach()
			# bs, num_queries = output_state.shape[:2]
			
			for batchID in range(batchSize):
				# alive_idx = output_logits[batchID, :].squeeze(-1) > existanceThreshold
				maxThreshold = max([torch.max(output_logits[batchID, :].squeeze(-1)), existanceThreshold])
				alive_idx = output_logits[batchID, :].squeeze(-1) == maxThreshold
				alive_output = output_state[batchID, alive_idx, :].cpu().numpy() 	# Pred
				current_targets = labels[batchID].cpu().numpy()						# Truth

				if np.size(alive_output) == 0:
					xDiff = yDiff = cutoffDistance**2
				else:
					xDiff = min((alive_output[0, 0] - current_targets[0, 0])**2, cutoffDistance**2)
					yDiff = min((alive_output[0, 1] - current_targets[0, 1])**2, cutoffDistance**2)
				# end if
				xRMSE[nCycle, batchID] += xDiff
				yRMSE[nCycle, batchID] += yDiff
			# end for
		# end for
		nEpoch += 1
	# end while
	xRMSE = np.sqrt(xRMSE/epoch)
	yRMSE = np.sqrt(yRMSE/epoch)
	return xRMSE, yRMSE

def SeqPredExport(dataGenerator, model, winStep, epoch, existanceThreshold = 0.9):
	'''
	序列化预测结果导出
	'''
	outputTotal = []
	for epochID in range(epoch):
		dataGenerator.SetEpoch(epochID)
		outputEpoch = []
		for stepID in range(winStep):
			# batch, labels, unique_ids = next(dataGenerator)
			batch, labels, unique_ids = dataGenerator.Get_batch()
			output, _, _, _, _ = model.forward(batch, unique_ids)
			output_state = output['state'].detach()
			output_logits = output['logits'].sigmoid().detach()
			bs, num_queries = output_state.shape[:2]
			for batchID in range(bs):
				# alive_idx = output_logits[batchID, :].squeeze(-1) > existanceThreshold					# 多目标，选择所有置信度高于阈值的项
				maxThreshold = max([torch.max(output_logits[batchID, :].squeeze(-1)), existanceThreshold])	# 单目标，选择置信度高于阈值的最高项
				alive_idx = output_logits[batchID, :].squeeze(-1) == maxThreshold

				alive_output = output_state[batchID, alive_idx, :].cpu().numpy()
				outputEpoch.append(alive_output.T)
			# end for
		# end for
		outputTotal.append(outputEpoch)
	# end for
	return outputTotal

def SeqPredPlot(dataGenerator, model, timeStep, existanceThreshold = 0.9):
	'''
	序列化预测结果绘制
	'''
	plt.figure(figsize=(8, 8), dpi=300)
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False
	xLim = []; yLim = []
	for stepID in range(timeStep):
		# batch, labels, unique_ids = next(dataGenerator)
		batch, labels, unique_ids = dataGenerator.Get_batch()
		output, _, _, _, _ = model.forward(batch, unique_ids)
		output_state = output['state'].detach()
		output_logits = output['logits'].sigmoid().detach()
		bs, num_queries = output_state.shape[:2]
		for batchID in range(bs):
			if stepID == 0:
				falseMeas = batch.tensors[batchID][(unique_ids[batchID] == -1)].cpu()
				plt.scatter(falseMeas.T[0][:-1], falseMeas.T[1][:-1], color='k', marker='+', alpha=falseMeas.T[2][:-1]/falseMeas.T[2].max()/1.2)
				plt.scatter(falseMeas.T[0][-1], falseMeas.T[1][-1], color='k', marker='+', alpha=1/1.2, label='虚假量测')
			# end if
			# alive_idx = output_logits[batchID, :].squeeze(-1) > existanceThreshold
			maxThreshold = max([torch.max(output_logits[batchID, :].squeeze(-1)), existanceThreshold])
			alive_idx = output_logits[batchID, :].squeeze(-1) == maxThreshold
			alive_output = output_state[batchID, alive_idx, :].cpu()
			current_targets = labels[batchID].cpu()
			# pointAlpha = 1 - stepID / timeStep
			pointAlpha = stepID / timeStep
			if stepID == 0 and batchID == 0:
				plt.scatter(current_targets.T[0], current_targets.T[1], color='b', marker='+', alpha=pointAlpha/2, label = '真实航迹')
				plt.scatter(alive_output.T[0], alive_output.T[1], color='r', marker='x', alpha=pointAlpha/1.2, label = '多目标智能跟踪结果')
			else:
				plt.scatter(current_targets.T[0], current_targets.T[1], color='b', marker='+', alpha=pointAlpha/2)
				plt.scatter(alive_output.T[0], alive_output.T[1], color='r', marker='x', alpha=pointAlpha/1.2)
			# end if
			if stepID == 0 or stepID == timeStep - 1:
				xLim.append(current_targets.T[0])
				yLim.append(current_targets.T[1])
			# end if
		# end for
	# end for
	xLim.sort(); yLim.sort()
	plt.xlim((xLim[0]-10, xLim[1]+10))
	plt.ylim((yLim[0]-10, yLim[1]+10))
	plt.xlabel('X / m')
	plt.ylabel('Y / m')
	plt.legend(loc = 1)
	plt.grid(True, linestyle="--", color="k", linewidth=0.5, alpha=0.3)
	plt.show()

def PlotResult(batch, labels, output):
	'''
	训练数据绘制
	'''
	outputs = output
	targets = labels
	existance_threshold = 0.8

	output_state = outputs['state'].detach()
	output_logits = outputs['logits'].sigmoid().detach()
	bs, num_queries = output_state.shape[:2]

	# plt.subplot(5, 4, pnum)

	colorEnum = ['r', 'y', 'g', 'c', 'b', 'm', 'r', 'y', 'g', 'c', 'b', 'm']
	for measBatch in range(bs):
		measPosX = batch.tensors[measBatch, ~batch.mask[measBatch], 0].cpu()
		measPosY = batch.tensors[measBatch, ~batch.mask[measBatch], 1].cpu()
		measAlph = (batch.tensors[measBatch, ~batch.mask[measBatch], 2]/max(batch.tensors[measBatch, ~batch.mask[measBatch], 2])/2).cpu()
		# measAlph = 1 - fnum/19
		
		alive_idx = output_logits[measBatch, :].squeeze(-1) > existance_threshold
		alive_output = output_state[measBatch, alive_idx, :].cpu()
		current_targets = targets[measBatch].cpu()

		# plt.figure(figsize=(8, 8), dpi=300)
		# # 笛卡尔
		plt.scatter(measPosX, measPosY, color='k', marker='+', alpha=measAlph)
		plt.scatter(current_targets.T[0], current_targets.T[1], color='b', marker='+', alpha=0.5, label = 'Ground Truth')
		plt.scatter(alive_output.T[0], alive_output.T[1], color='r', marker='x', alpha=0.5, label = 'MT3 Prediction')

		# 极坐标
		# ax = plt.axes(projection='polar')
		# ax.scatter(measPosY*np.pi/180.0, measPosX, color='k', marker='+', alpha=measAlph)
		# ax.scatter(current_targets.T[1]*np.pi/180.0, current_targets.T[0], color='b', marker='+', alpha=0.5, label = 'Ground Truth')
		# ax.scatter(alive_output.T[1]*np.pi/180.0, alive_output.T[0], color='r', marker='x', alpha=0.5, label = 'MT3 Prediction')

		# for targetNo in list(trajectories[measBatch]):
		# 	trackPosX = trajectories[measBatch][targetNo].T[0]
		# 	trackPosY = trajectories[measBatch][targetNo].T[1]
		# 	trackAlph = trajectories[measBatch][targetNo].T[4]/max(trajectories[measBatch][targetNo].T[4])
		# 	plt.scatter(trackPosX, trackPosY, color=colorEnum[targetNo], marker='x', alpha=trackAlph)

		# 笛卡尔
		# plt.xlim((-100, 100))
		# plt.ylim((-100, 100))
		plt.grid(True, linestyle="--", color="k", linewidth=0.5, alpha=0.3)
		plt.legend(loc=1, frameon=True, fontsize = 8)
		plt.xlabel('Range / m')
		plt.ylabel('Angle / °')
		# 极坐标
		# ax.set_thetagrids(np.arange(-90.0, 90.0, 10.0))
		# # ax.set_rgrids(np.arange(0, 100, 10))
		# ax.grid(True, linestyle="--", color="k", linewidth=0.5, alpha=0.3)
		# ax.set_axisbelow('True')
		# plt.legend(loc=1, frameon=True)
		# plt.xlim((-np.pi/2, np.pi/2))
		# # plt.ylim((0, 100))

		plt.show()