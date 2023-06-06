# %%
import multiprocessing
import argparse
import itertools

import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from util.load_config_files import load_yaml_into_dotdict
from util.misc import NestedTensor

# %%
def main():
	# 从CLI载入yaml
	parser = argparse.ArgumentParser()
	parser.add_argument('--task_params', default='./yaml/task.yaml')
	parser.add_argument('--model_params', default='./yaml/mt3.yaml')
	args = parser.parse_known_args()[0]
	print(f'Task configuration file: {args.task_params}')
	print(f'Model configuration file: {args.model_params}')

	# 从yaml载入超参数
	params = load_yaml_into_dotdict(args.task_params)
	params.update(load_yaml_into_dotdict(args.model_params))

	# 生成训练数据
	data_generator = DataGenerator(params)
	batch, labels, unique_ids, trajectories = data_generator.get_batch()

	# 训练数据绘制
	colorEnum = ['r', 'y', 'g', 'c', 'b', 'm', 'r', 'y', 'g', 'c', 'b', 'm']
	batchRange = [2]
	for measBatch in batchRange:
		measPosX = batch.tensors[measBatch, ~batch.mask[measBatch], 0]
		measPosY = batch.tensors[measBatch, ~batch.mask[measBatch], 1]
		measAlph = batch.tensors[measBatch, ~batch.mask[measBatch], 2]/max(batch.tensors[measBatch, ~batch.mask[measBatch], 2])
		plt.scatter(measPosX, measPosY, color='k', s=20, marker='.', alpha=measAlph)
		# plt.scatter(labels[measBatch].T[0], labels[measBatch].T[1], color='b', marker='x', alpha=1)
		for targetNo in list(trajectories[measBatch]):
			trackPosX = trajectories[measBatch][targetNo].T[0]
			trackPosY = trajectories[measBatch][targetNo].T[1]
			trackAlph = trajectories[measBatch][targetNo].T[4]/max(trajectories[measBatch][targetNo].T[4])
			plt.scatter(trackPosX, trackPosY, color=colorEnum[targetNo], marker='x', alpha=trackAlph)

		plt.xlim((-10, 10))
		plt.ylim((-10, 10))
		# plt.show()
		plt.savefig("training_data.png")

# %%
def GetWinBatch(params):
	data_generator = DataGenerator(params)
	timeSteps = params.data_generation.n_timesteps
	windowSize = params.data_generation.n_windowsize
	batchSize = params.training.batch_size
	device = params.training.device
	dt = params.data_generation.dt

	while True:

		batch, labels, unique_ids, trajectories = data_generator.get_batch()
		mixBatch = torch.cat((batch.mask.unsqueeze(2), unique_ids.unsqueeze(2), batch.tensors), dim=2) # mask, unique_ids, tensors(x, y, t)
		mixBatch[:, :, 4] = (mixBatch[:, :, 4] / dt + 0.5).type(torch.int) # 时间步取整
		mixBatch[:, :, 4][mixBatch[:, :, 0] == 1] = -1 # 排除masked时间步

		for windowId in range(timeSteps - windowSize, -1, -1):
			windowMask = (mixBatch[:, :, 4] > (windowId - 0.5)) & (mixBatch[:, :, 4] < (windowId + windowSize - 0.5)) # 有效数据Mask
			batchLength = windowMask.sum(dim=1).max().item()
			windowSelected = torch.zeros((batchSize, batchLength, 5), dtype=torch.float32, device=device) # 窗口mixBatch
			windowSelected[:, :, 0] = 1
			windowSelected[:, :, 1] = -2
			for batchId in range(batchSize):
				windowSelected[batchId, :windowMask[batchId].sum()] = mixBatch[batchId, windowMask[batchId]]
				# labels处理
				label = np.zeros((0, 2))
				for trackTarget in trajectories[batchId].values():
					label = np.vstack((label, trackTarget[windowId + windowSize - 1, :2]))
					
				labels[batchId] = Tensor(label).to(device)

			windowSelected[:, :, 4][windowSelected[:, :, 4].type(torch.int) != 0] -= windowId
			windowSelected[:, :, 4] *= dt

			outBatch = NestedTensor(windowSelected[:, :, 2:5],
									windowSelected[:, :, 0].type(torch.int).type(torch.bool))

			outUnique_ids = windowSelected[:, :, 1]

			yield outBatch, labels, outUnique_ids

		batch = outBatch
		batchLen = batch.tensors[0, :, 2].shape[0]
		unique_ids = outUnique_ids
		for trackBatch in trajectories:
			for targetId, trackTarget in trackBatch.items():
				trackBatch[targetId] = trackTarget[trackTarget[:, 4] < (windowSize-0.5)*dt]

		for _ in range(windowSize - 2):
			for batchId in range(batchSize):
				# batch.tensors与unique_ids同步处理
				maxTimeStep = batch.tensors[batchId, :, 2].max()	# 最大时间插值
				maxMeasSum = (batch.tensors[batchId, :, 2] == maxTimeStep).sum().item()	# 最大时间插值个数：最大时间步的量测数
				maxMeasId = ((batch.tensors[batchId, :, 2] == maxTimeStep) + 0).argmax(dim=0).item()	# 最大时间插值索引
				# 标记包含最大时间插值的行
				mask = torch.ones(batchLen).bool()
				mask[maxMeasId:maxMeasId+maxMeasSum] = False
				# 删除包含最大时间插值的行
				tensorsTemp = batch.tensors[batchId, mask, :]	# tensors
				uniqidsTemp = unique_ids[batchId, mask]	# unique_ids

				# 序列前补0，时间插值递增
				tensorsTemp[:maxMeasId, 2] += 0.1
				tensorsTemp = torch.cat((torch.zeros([maxMeasSum, 3], device=device), tensorsTemp), 0)
				batch.tensors[batchId] = tensorsTemp
				uniqidsTemp = torch.cat((torch.zeros(maxMeasSum, device=device)-2, uniqidsTemp), 0)
				unique_ids[batchId] = uniqidsTemp

				# batch.mask处理 # NOTICE mask与tensor非同步处理，mask没有移位操作，仅与补0保持数量相等
				maxMaskId = (~batch.mask[batchId] + 0).argmax(dim=0).item()	# 第一个False索引
				batch.mask[batchId, maxMaskId:maxMaskId+maxMeasSum] = True	# 从False索引开始连续mask maxMeasSum个

				# labels处理
				label = np.zeros((0, 2))
				for targetId in list(trajectories[batchId]):
					trajectories[batchId][targetId] = np.delete(trajectories[batchId][targetId], -1, axis=0)
					label = np.vstack((label, trajectories[batchId][targetId][-1, :2]))
					
				labels[batchId] = Tensor(label).to(device)
				
			yield batch, labels, unique_ids
			
# %%
def GetSeqBatch(params):
	data_generator = DataGenerator(params)
	n_timesteps = params.data_generation.n_timesteps - 1
	batch_size = params.training.batch_size
	device = params.training.device
	resetCount = 0

	while True:

		if resetCount != 0:
		# if False:
			for batchId in np.arange(batch_size):
				# batch.tensors与unique_ids同步处理
				maxTimeStep = batch.tensors[batchId, :, 2].max()	# 最大时间插值
				maxMeasSum = (batch.tensors[batchId, :, 2] == maxTimeStep).sum().item()	# 最大时间插值个数：最大时间步的量测数
				maxMeasId = ((batch.tensors[batchId, :, 2] == maxTimeStep) + 0).argmax(dim=0).item()	# 最大时间插值索引
				# 标记包含最大时间插值的行
				mask = torch.ones(batchLen).bool()
				mask[maxMeasId:maxMeasId+maxMeasSum] = False
				# 删除包含最大时间插值的行
				tensorsTemp = batch.tensors[batchId, mask, :]	# tensors
				uniqidsTemp = unique_ids[batchId, mask]	# unique_ids

				# 序列前补0，时间插值递增
				tensorsTemp[:maxMeasId, 2] += 0.1
				tensorsTemp = torch.cat((torch.zeros([maxMeasSum, 3], device=device), tensorsTemp), 0)
				batch.tensors[batchId] = tensorsTemp
				uniqidsTemp = torch.cat((torch.zeros(maxMeasSum, device=device)-2, uniqidsTemp), 0)
				unique_ids[batchId] = uniqidsTemp

				# batch.mask处理 # NOTICE mask与tensor非同步处理，mask没有移位操作，仅与补0保持数量相等
				maxMaskId = (~batch.mask[batchId] + 0).argmax(dim=0).item()	# 第一个False索引
				batch.mask[batchId, maxMaskId:maxMaskId+maxMeasSum] = True	# 从False索引开始连续mask maxMeasSum个

				# labels处理
				label = np.zeros((0, 2))
				for targetId in list(trajectories[batchId]):
					trajectories[batchId][targetId] = np.delete(trajectories[batchId][targetId],-1,axis=0)
					label = np.vstack((label, trajectories[batchId][targetId][-1, :2]))
					
				label = Tensor(label).to(device)
				labels[batchId] = label

		else:
			resetCount = n_timesteps
			batch, labels, unique_ids, trajectories = data_generator.get_batch()
			batchLen = batch.tensors[0, :, 2].shape[0]
			
		resetCount -= 1
		yield batch, labels, unique_ids
# %%
class Object:

	def __init__(self, pos, vel, t, delta_t, sigma, id):
		self.pos = pos
		self.vel = vel
		self.delta_t = delta_t
		self.sigma = sigma
		self.state_history = np.array([np.concatenate([pos,vel,np.array([t])])])
		self.process_noise_matrix = sigma*np.array([[delta_t ** 3 / 3, delta_t ** 2 / 2], [delta_t ** 2 / 2, delta_t]])

		# Unique identifier for every object
		self.id = id

	def update(self, t, rng):
		"""
		Updates this object's state using a discretized constant velocity model.
		"""

		# Update position and velocity of the object in each dimension separately
		assert len(self.pos) == len(self.vel)
		process_noise = rng.multivariate_normal([0, 0], self.process_noise_matrix, size=len(self.pos))
		self.pos += self.delta_t * self.vel + process_noise[:,0]
		self.vel += process_noise[:,1]

		# Add current state to previous states
		self.state_history = np.vstack((self.state_history,np.concatenate([self.pos.copy(),self.vel.copy(),np.array([t])])))

	def __repr__(self):
		return 'id: {}, pos: {}, vel: {}'.format(self.id, self.pos, self.vel)

class MotDataGenerator:
	def __init__(self, args, rng):
		self.start_pos_params = [args.data_generation.mu_x0, args.data_generation.std_x0]
		self.start_vel_params = [args.data_generation.mu_v0, args.data_generation.std_v0]
		self.prob_add_obj = args.data_generation.p_add
		self.prob_remove_obj = args.data_generation.p_remove
		self.delta_t = args.data_generation.dt
		self.process_noise_intens = args.data_generation.sigma_q
		self.prob_measure = args.data_generation.p_meas
		self.measure_noise_intens = args.data_generation.sigma_y
		self.n_average_false_measurements = args.data_generation.n_avg_false_measurements
		self.n_average_starting_objects = args.data_generation.n_avg_starting_objects
		self.field_of_view_lb = args.data_generation.field_of_view_lb
		self.field_of_view_ub = args.data_generation.field_of_view_ub
		self.birth_lb = args.data_generation.birth_lb
		self.birth_ub = args.data_generation.birth_ub
		self.max_objects = args.data_generation.max_objects
		self.rng = rng
		self.dim = len(self.start_pos_params[0])

		self.debug = False
		assert self.n_average_starting_objects != 0, 'Datagen does not currently work with n_avg_starting_objects equal to zero.'

		self.t = None
		self.objects = None
		self.trajectories = None
		self.measurements = None
		self.unique_ids = None
		self.unique_id_counter = None
		self.reset()

	def reset(self):
		self.t = 0
		self.objects = []
		self.trajectories = {}
		self.measurements = np.array([])
		self.unique_ids = np.array([], dtype='int64')
		self.unique_id_counter = itertools.count()

		# Add initial set of objects (re-sample until we get a nonzero value)
		n_starting_objects = 0
		while n_starting_objects == 0:
			n_starting_objects = self.rng.poisson(self.n_average_starting_objects)
		self.add_objects(n_starting_objects)

		# Measure the initial set of objects
		self.generate_measurements()

		if self.debug:
			print(n_starting_objects, 'starting objects')

	def create_new_object(self, pos, vel):
		return Object(pos=pos,
					  vel=vel,
					  t=self.t,
					  delta_t=self.delta_t,
					  sigma=self.process_noise_intens,
					  id=next(self.unique_id_counter))

	def add_objects(self, n):
		"""
		Adds `n` new objects to `objects` list.
		"""
		# Never add more objects than the maximum number of allowed objects
		n = min(n, self.max_objects-len(self.objects))
		if n == 0:
			return

		# Create new objects and save them in the datagen
		positions = self.rng.uniform(low=self.birth_lb, high=self.birth_ub, size=(n,self.dim))
		velocities = self.rng.multivariate_normal(self.start_vel_params[0], self.start_vel_params[1], size=(n,))
		self.objects += [self.create_new_object(pos, vel) for pos,vel in zip(positions, velocities)]

	def remove_far_away_objects(self):
		if len(self.objects) == 0:
			return

		positions = np.array([obj.pos for obj in self.objects])
		lb = positions < self.field_of_view_lb
		ub = positions > self.field_of_view_ub
		remove_elements = np.bitwise_or(lb.any(axis=1), ub.any(axis=1))

		self.objects = [o for o, r in zip(self.objects, remove_elements) if not r]

	def remove_objects(self, p):
		"""
		Removes each of the objects with probability `p`.
		"""

		# Compute which objects are removed in this time-step
		deaths = self.rng.binomial(n=1, p=p, size=len(self.objects))

		n_deaths = sum(deaths)
		if self.debug and (n_deaths > 0):
			print(n_deaths, 'objects were removed')

		# Save the trajectories of the removed objects
		for obj, death in zip(self.objects, deaths):
			if death:
				self.trajectories[obj.id] = obj.state_history

		# Remove them from the object list
		self.objects = [o for o, d in zip(self.objects, deaths) if not d]

	def get_prob_death(self, obj):
		return self.prob_remove_obj

	def remove_object(self, obj, p = None):
		"""
		Removes an object based on its state
		"""
		if p is None:
			p = self.get_prob_death(obj)

		r = self.rng.rand()

		if r < p:
			return True
		else:
			return False

	def generate_measurements(self):
		"""
		Generates all measurements (true and false) for the current time-step.
		"""
		# Generate the measurement for each object with probability `self.prob_measure`
		is_measured = self.rng.binomial(n=1, p=self.prob_measure, size=len(self.objects))
		measured_objects = [obj for obj, is_measured in zip(self.objects, is_measured) if is_measured]
		measurement_noise = self.rng.normal(0, self.measure_noise_intens, size=(len(measured_objects),self.dim))
		true_measurements = np.array([np.append(obj.pos+noise, self.t) for obj, noise in zip(measured_objects, measurement_noise)])

		# Generate false measurements
		n_false_measurements = self.rng.poisson(self.n_average_false_measurements)
		false_meas = self.rng.uniform(self.field_of_view_lb, self.field_of_view_ub, size=(n_false_measurements,self.dim))
		false_measurements = np.ones((n_false_measurements,self.dim+1)) * self.t
		false_measurements[:,:-1] = false_meas

		# Also save from which object each measurement came from (for contrastive learning later); -1 is for false meas.
		unique_obj_ids_true = [obj.id for obj in measured_objects]
		unique_obj_ids_false = [-1]*len(false_measurements)
		unique_obj_ids = np.array(unique_obj_ids_true + unique_obj_ids_false)

		# Concatenate true and false measurements in a single array
		if true_measurements.shape[0] and false_measurements.shape[0]:
			new_measurements = np.vstack([true_measurements, false_measurements])
		elif true_measurements.shape[0]:
			new_measurements = true_measurements
		elif false_measurements.shape[0]:
			new_measurements = false_measurements
		else:
			return

		# Shuffle all generated measurements and corresponding unique ids in unison
		random_idxs = self.rng.permutation(len(new_measurements))
		new_measurements = new_measurements[random_idxs]
		unique_obj_ids = unique_obj_ids[random_idxs]

		# Save measurements and unique ids
		self.measurements = np.vstack([self.measurements, new_measurements]) if self.measurements.shape[0] else new_measurements
		self.unique_ids = np.hstack([self.unique_ids, unique_obj_ids])

	def step(self, add_new_objects=True):
		"""
		Performs one step of the simulation.
		"""
		self.t += self.delta_t

		# Update the remaining ones
		for obj in self.objects:
			obj.update(self.t, self.rng)

		# Remove objects that left the field-of-view
		self.remove_far_away_objects()

		# Add new objects
		if add_new_objects:
			n_new_objs = self.rng.poisson(self.prob_add_obj)
			self.add_objects(n_new_objs)

		# Remove some of the objects
		self.remove_objects(self.prob_remove_obj)
		
		# Generate measurements
		self.generate_measurements()
		
		if self.debug:
			if n_new_objs > 0:
				print(n_new_objs, 'objects were added')
			print(len(self.objects))

	def finish(self):
		"""
		Should be called after the last call to `self.step()`. Removes the remaining objects, consequently adding the
		remaining parts of their trajectories to `self.trajectories`.
		"""
		self.remove_objects(1.0)

class DataGenerator:
	def __init__(self, params):
		self.params = params
		assert 0 <= params.data_generation.n_prediction_lag <= params.data_generation.n_timesteps, "Prediction lag has to be smaller than the total number of time-steps."
		self.device = params.training.device
		self.n_timesteps = params.data_generation.n_timesteps

		self.pool = multiprocessing.Pool()

		# Create `batch_size` data generators, each with its own independent (to a high probability) RNG
		ss = SeedSequence(params.data_generation.seed)
		rngs = [default_rng(s) for s in ss.spawn(params.training.batch_size)]
		self.datagens = [MotDataGenerator(params, rng=rng) for rng in rngs]

	def get_batch(self):
		results = self.pool.starmap(get_single_training_example, zip(self.datagens, [self.n_timesteps]*len(self.datagens)))

		# Unpack results
		training_data, labels, unique_ids, trajectories, new_rngs = tuple(zip(*results))
		labels = [Tensor(l).to(torch.device(self.device)) for l in labels]
		trajectories = list(trajectories)
		unique_ids = [list(u) for u in unique_ids]

		# Update the RNGs of all the datagens for next call
		for datagen, new_rng in zip(self.datagens, new_rngs):
			datagen.rng = new_rng

		# Pad training data
		max_len = max(list(map(len, training_data)))
		training_data, mask = pad_to_batch_max(training_data, max_len)

		# Pad unique ids
		for i in range(len(unique_ids)):
			unique_id = unique_ids[i]
			n_items_to_add = max_len - len(unique_id)
			unique_ids[i] = np.concatenate([unique_id, [-2] * n_items_to_add])[None, :]
		unique_ids = np.concatenate(unique_ids)

		training_nested_tensor = NestedTensor(Tensor(training_data).to(torch.device(self.device)),
											  Tensor(mask).bool().to(torch.device(self.device)))
		unique_ids = Tensor(unique_ids).to(self.device)

		return training_nested_tensor, labels, unique_ids, trajectories

def pad_to_batch_max(training_data, max_len):
	batch_size = len(training_data)
	d_meas = training_data[0].shape[1]
	training_data_padded = np.zeros((batch_size, max_len, d_meas))
	mask = np.ones((batch_size, max_len))
	for i, ex in enumerate(training_data):
		training_data_padded[i,:len(ex),:] = ex
		mask[i,:len(ex)] = 0

	return training_data_padded, mask

def get_single_training_example(data_generator, n_timesteps):
	"""Generates a single training example

	Returns:
		training_data   : A single training example
		true_data       : Ground truth for example
	"""

	data_generator.reset()
	label_data = []

	while len(label_data) == 0 or len(data_generator.measurements) == 0:

		# Generate n_timesteps of data, from scratch
		data_generator.reset()
		for i in range(n_timesteps - 1):
			add_new_objects_flag = i < n_timesteps -3  # don't add new objects in the last two timesteps of generation, for cleaner training labels
			data_generator.step(add_new_objects=add_new_objects_flag)
		data_generator.finish()

		# -1 is applied because we count t=0 as one time-step
		for traj_id in data_generator.trajectories:
			traj = data_generator.trajectories[traj_id]
			if round(traj[-1][-1] / data_generator.delta_t) == n_timesteps - 1: #last state of trajectory, time
				pos = traj[-1][:data_generator.dim].copy()
				label_data.append(pos)

	training_data = np.array(data_generator.measurements.copy())
	unique_measurement_ids = data_generator.unique_ids.copy()
	new_rng = data_generator.rng

	return training_data, np.array(label_data), unique_measurement_ids, data_generator.trajectories.copy(), new_rng

# %%
if __name__ == '__main__':
	main()
