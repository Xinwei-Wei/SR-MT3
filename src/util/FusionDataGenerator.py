from __future__ import annotations
import yaml
import numpy as np
import multiprocessing

class PlatformModel:
	''' Model of Platforms.

	'''
	def __init__(self, platformArg:dict):
		self.is2D = platformArg['is2D']
		self.T = platformArg['T']
		self.t0 =  platformArg['t0']
		self.tf = platformArg['tf']
		
		self.N_scan = int((self.tf * 1000 - self.t0) / self.T + 1)
		self.F = np.block( [ [np.eye(3),		np.eye(3)*(self.T/1000.0)], \
							 [np.zeros([3, 3]),	np.eye(3)] ] )
		q = np.array(platformArg['q'])
		self.Q = np.block( [ [np.diag(q)*(self.T/1000)**3/3.0,	np.diag(q)*(self.T/1000)**2/2.0], \
							 [np.diag(q)*(self.T/1000)**2/2.0,	np.diag(q)*(self.T/1000)] ] )
		self.X0 = np.array(platformArg['X0'])
		self.area = np.array(platformArg['area'])
		self.area[1:3, :] = self.area[1:3, :] * np.pi / 180.0

class TargetModel:
	''' Model of Targets.

	'''
	def __init__(self, targetArg:dict):
		self.is2D = targetArg['is2D']
		self.T = targetArg['T']
		self.t0 = targetArg['t0']
		self.tf = targetArg['tf']
		self.numAvg = targetArg['numAvg']
		self.numSig = targetArg['numSig']
		self.lifetimeAvg = targetArg['lifetimeAvg']
		self.lifetimeSig = targetArg['lifetimeSig']
		
		self.N_scan = int((self.tf * 1000 - self.t0) / self.T + 1)
		self.F = np.block( [ [np.eye(3),		np.eye(3)*(self.T/1000.0)], \
							 [np.zeros([3, 3]),	np.eye(3)] ] )
		q = np.array(targetArg['q'])
		self.Q = np.block( [ [np.diag(q)*(self.T/1000)**3/3.0,	np.diag(q)*(self.T/1000)**2/2.0], \
							 [np.diag(q)*(self.T/1000)**2/2.0,	np.diag(q)*(self.T/1000)] ] )
		self.num = np.around(np.random.normal(self.numAvg, self.numSig)).astype(np.int16)
		self.num = max(1, self.num)	# TODO 保证至少有一个目标
		birthRange = np.array(targetArg['birthRange'])
		vAbsRange = np.array(targetArg['vAbsRange'])
		xstart = np.vstack( (np.expand_dims(birthRange[:, 0], axis=1) + np.random.rand(3, self.num) * np.expand_dims((birthRange[:, 1] - birthRange[:, 0]), axis=1), \
							 np.sign(np.random.randn(3, self.num)) * (np.expand_dims(vAbsRange[:, 0], axis=1) + np.random.rand(3, self.num) * np.expand_dims((vAbsRange[:, 1] - vAbsRange[:, 0]), axis=1))) )
		tbirth = self.t0 + (self.tf - self.t0) * np.random.uniform(targetArg['timeRange'][0], targetArg['timeRange'][1], [1, self.num])
		tdeath = tbirth + np.random.normal(self.lifetimeAvg, self.lifetimeSig, [1, self.num])
		self.X0 = xstart
		self.lifetime = np.vstack( (tbirth, tdeath) )

class SensorModel:
	''' Model of Sensors' Observation.

	'''
	def __init__(self, sensorArg:dict):
		self.type = np.zeros(4)
		for s, arg in sensorArg.items():
			if s == 'Radar':
				self.type[0] = 1
				self.Radar = dict()
				sensor = self.Radar
				sensor['h'] = self.RadarH

			if s == 'Pradar':
				self.type[1] = 1
				self.Pradar = dict()
				sensor = self.Pradar
				sensor['h'] = self.PradarH

			if s == 'DAS':
				self.type[2] = 1
				self.DAS = dict()
				sensor = self.DAS
				sensor['h'] = np.array([[0, 0, 0, 1, 0, 0],
										[0, 0, 0, 0, 1, 0]])
				
			if s == 'Rect':
				self.type[3] = 1
				self.Rect = dict()
				sensor = self.Rect
				sensor['h'] = np.array([[1, 0, 0, 0, 0, 0],
			    						[0, 1, 0, 0, 0, 0],
										[0, 0, 1, 0, 0, 0]])

			sensor['Pd'] = arg['Pd']
			sensor['T'] = arg['T']
			sensor['lmd'] = arg['lmd']
			sensor['R'] = np.diag(np.array(arg['R']).astype(np.float32))
			sensor['area'] = np.array(arg['area'])
			if s != 'Rect':
				sensor['area'][-2:-1, :] = sensor['area'][-2:-1, :] * np.pi / 180.0
			
	def RadarH(self, p_s:np.ndarray, p_t:np.ndarray):
		return np.vstack( (np.linalg.norm(p_t-p_s), \
						   np.arctan2(p_t[1]-p_s[1], p_t[0]-p_s[0]), \
						   np.arctan((p_t[2]-p_s[2]) / np.linalg.norm(p_t[0:1]-p_s[0:1]))) )
	def PradarH(self, p_s:np.ndarray, p_t:np.ndarray):
		return np.vstack( (np.arctan2(p_t[1]-p_s[1], p_t[0]-p_s[0]), \
						   np.arctan((p_t[2]-p_s[2]) / np.linalg.norm(p_t[0:1]-p_s[0:1]))) )

class FusionDataGenerator:
	''' Generates Simulation Data of Multi-Sensors.

	'''
	def __init__(self, filepath:str, evalBS = -1):
		self.__filepath = filepath
		self.params = self.__ReadParams()
		if evalBS == -1:
			self.__batchSize:int = self.params['totalArg']['batchSize']
		else:
			self.__batchSize = evalBS
		self.pool = multiprocessing.Pool(self.__batchSize)

	def __ReadParams(self) -> dict:
		''' Read Parameters from yaml file.

		'''
		with open(self.__filepath, 'r') as f:
			try:
				return yaml.safe_load(f)
			except yaml.YAMLError as exc:
				print(f"Error loading yaml file. Error: {exc}")
				exit()

	def GenMeasurements(self):
		''' Generates Multi-Sensors' Obervation for all Batches.

			:returns: batches[sensors{sensorType, meas{t/Z/Z_Truth, value}}] where Z/Z_Truth as [ids; r(only radar); theta; phi]
		'''
		# 写入模型
		self.GetModels()
		# # 串行计算所有batch
		# Meas = []
		# for n in range(self.__batchSize):
		# 	mea = self.GenMeasurementsK(self.pModel[n], self.tModel[n], self.sModel[n])
		# 	Meas.append(mea)
		# return Meas
		# 并行计算所有batch
		return self.pool.starmap(self.GenMeasurementsK, zip(self.pModel, self.tModel, self.sModel))

	def GetModels(self):
		''' Generates Platforms, Targets and Sensors Model based on Params.

		'''
		self.pModel = [PlatformModel(self.params['platformArg']) for i in range(self.__batchSize)]
		self.tModel = [TargetModel(self.params['targetArg']) for i in range(self.__batchSize)]
		self.sModel = [SensorModel(self.params['sensorArg']) for i in range(self.__batchSize)]

	def GenMeasurementsK(self, pModel:PlatformModel, tModel:TargetModel, sModel:SensorModel):
		''' Generates Multi-Sensors' Obervation for one Batch.

			:returns: sesors{sensorType, meas{type, value}}
		'''
		# 生成真实航迹
		pTruth = self.__GenTruth(pModel, 'platform')
		tTruth = self.__GenTruth(tModel, 'target')
		# 生成相对航迹
		xi = self.__targetMSC(tModel.N_scan, tTruth['X'], pTruth['X'])
		# 生成各传感器量测
		Meas = self.__GenObservationTotal(tModel, sModel, pTruth, tTruth, xi)
		return Meas
	
	def __GenObservationTotal(self, tModel:TargetModel, sModel:SensorModel, pTruth:dict, tTruth:dict, xi:list[np.ndarray]):
		''' Generates Obervation for all Sensors.

			:param tModel: TargetModel
			:param sModel: SensorModel
			:param pTruth: ture-track of platforms
			:param tTruth: ture-track of targets
			:param xi: relative measurement in MSC
			:returns: obervation of each sensor in a dict
		'''
		Obs = dict({'Radar': dict(), 'Pradar': dict(), 'DAS': dict(), 'Rect': dict()})
		is2D = tModel.is2D
		for n in range(4):
			if sModel.type[n] == 0:
				continue
			# end if
			if n == 0:
				s = 'Radar'
				arg = sModel.Radar
			if n == 1:
				s = 'Pradar'
				arg = sModel.Pradar
			if n == 2:
				s = 'DAS'
				arg = sModel.DAS
			if n == 3:
				s = 'Rect'
				arg = sModel.Rect

			K_n = int(np.floor((tModel.tf*1000 - tModel.t0) / arg['T']) + 1)	# 量测产生总帧数
			Z = []
			Z_Truth = []
			t = np.zeros([K_n])
			for k in range(K_n):
				t[k] = (tModel.t0 + k*arg['T']) / 1000.0;						# 量测产生时刻
				k_t = np.nonzero(tTruth['t'] == t[k])[0][0];					# 该时刻对应目标的帧数
				Z_k, Z_Truth_k = self.__GenObservationK(tTruth['X'][k_t], tTruth['ids'][k_t], xi[k_t], pTruth['X'][:,[k_t]], s, arg, is2D)
				Z.append(Z_k)
				Z_Truth.append(Z_Truth_k)
			# end for
			Obs[s]['t'] = t
			Obs[s]['Z'] = Z
			Obs[s]['Z_Truth'] = Z_Truth
		# end for
		return Obs

	def __GenObservationK(self, Xt:np.ndarray, id:np.ndarray, Xi:np.ndarray, Xs:np.ndarray, type:str, sensor:dict, is2D:bool) -> tuple[np.ndarray, np.ndarray]:
		''' Generates Obervation for one Sensor.

			:param Xt: ture-track of targets
			:param id: targets' id in ture-track
			:param Xi: relative measurement in MSC
			:param Xs: ture-track of platforms
			:param type: sensor type
			:param sensor: sensor parameters
			:param is2D: if the scenario is 2D
			:returns: Observation Z, Truth Z_Truth as [ids; r(only radar); theta; phi]
		'''
		Z = np.empty([sensor['R'].shape[0], 0])
		Z_Truth = np.empty([sensor['R'].shape[0], 0])
		ids = np.empty([1, 0]).astype(np.int16)
		num_x = Xt.shape[1]
		Pd = sensor['Pd']
		h = sensor['h']
		R = sensor['R']
		lmd = sensor['lmd']
		area = sensor['area'][:, [1]] - sensor['area'][:, [0]]
		
		if num_x > 0 :
			
			# 平台运动计算
			p_speed = Xs[3:6, 0]
			if is2D:
				platform_epsilon = 0;																# 平台俯仰角计算(二维)
			else:
				platform_epsilon = np.arctan(p_speed[2] / np.sqrt(p_speed[0]**2 + p_speed[1]**2));	# 平台俯仰角计算
			# end if
			platform_beta = np.arctan2(p_speed[1], p_speed[0])										# 平台方位角计算
			p_s = Xs[0:3]

			# 产生每个目标的量测
			for i in range(num_x):
				p_t = Xt[0:3, i]
				if type == 'Radar':
					if np.random.rand() <= Pd:
						z = h(p_s, p_t)
						if z[0] >= sensor['area'][0, 0] and z[0] <= sensor['area'][0, 1] \
								and self.__AngleConfirm(z[1], platform_beta, area[1]/2) \
								and self.__AngleConfirm(z[2], platform_epsilon, area[2]/2):
							Z = np.c_[Z, z]
							ids = np.c_[ids, id[0, i]]
				# end if

				if type == 'Pradar':
					if np.random.rand() <= Pd:
						z = h(p_s, p_t)
						if self.__AngleConfirm(z[0], platform_beta, area[0]/2) \
								and self.__AngleConfirm(z[1], platform_epsilon, area[1]/2):
							Z = np.c_[Z, z]
							ids = np.c_[ids, id[0, i]]
				# end if

				if type == 'DAS':
					if np.random.rand() <= Pd:
						p_t = Xi[:, i]
						z = h @ p_t
						if self.__AngleConfirm(z[0], platform_beta, area[0]/2) \
								and self.__AngleConfirm(z[1], platform_epsilon, area[1]/2):
							Z = np.c_[Z, z]
							ids = np.c_[ids, id[0, i]]
				# end if

				if type == 'Rect':
					if np.random.rand() <= Pd:
						p_t = Xt[:, [i]] - Xs
						z = h @ p_t
						if z[0] >= sensor['area'][0, 0] and z[0] <= sensor['area'][0, 1] \
						   and z[1] >= sensor['area'][1, 0] and z[1] <= sensor['area'][1, 1] \
						   and z[2] >= sensor['area'][2, 0] and z[2] <= sensor['area'][2, 1]:
							Z = np.c_[Z, z]
							ids = np.c_[ids, id[0, i]]
				# end if
			# end for

			Z_Truth = np.r_[ids, Z.copy()]

			# 叠加噪声
			if np.size(Z) != 0:
				Z_noise = np.sqrt(R) @ np.random.randn(Z.shape[0], Z.shape[1])
				if is2D:
					Z_noise[-1, :] = 0
				# end if
				Z = Z + Z_noise
			# end if
		# end if

		# 产生杂波
		N_clutter = np.random.poisson(lmd)
		# if N_clutter + Z.shape[1] > lmd*2:
		# 	N_clutter = lmd*2 - Z.shape[1]
		# # end if
		Z_clutter = np.zeros([area.shape[0], N_clutter])
		for i in range(N_clutter):
			Z_clutter[:, [i]] = sensor['area'][:, [0]] + area * np.random.rand(area.shape[0], 1)
		# end for

		Z = np.c_[Z, Z_clutter]														# 加入杂波
		Z = np.r_[np.c_[ids, np.zeros([1, Z.shape[1] - ids.shape[1]]) - 1], Z]		# ids杂波补-1
		Z = Z[:, np.random.permutation(np.arange(Z.shape[1]))]						# 乱序

		return Z, Z_Truth

	def __AngleConfirm(self, a:float, b:float, c:float):
		''' Check whether the Measurement Angle is in the Detection Range.

			:param a: angle under test
			:param b: datum angle
			:param c: angle range
			:returns: if the distance between a & b is in the range of c
		'''
		beta_1 = np.arccos(np.cos(a-b))
		beta_2 = np.arcsin(np.sin(a-b))
		if beta_1 < np.pi/2 and beta_2 < 0:
			t = -beta_2
		elif beta_1 < np.pi/2 and beta_2 >= 0:
			t = beta_2
		elif beta_1 >= np.pi/2 and beta_2 >= 0:
			t = beta_1
		elif beta_1 >= np.pi/2 and beta_2 < 0:
			t = beta_1
		else:
			t=0
		# end if
		
		if t <= c:
			return True
		else:
			return False

	def __targetMSC(self, N_scan:int, X:list[np.ndarray], Xp:list[np.ndarray]):
		''' Calucates Relative Measurement between Platforms and Targets in MSC.

			:param N_scan: number of frames
			:param X: ture-track of targets
			:param Xp: ture-track of platforms
			:returns: N_scan[Relative Measurement(6 x Target Numbers)]
		'''
		xi = [np.zeros([6, 0]) for i in range(N_scan)]
		for k in range(N_scan):
			if X[k].size != 0:
				for i in range(X[k].shape[1]):
					Xr_k = X[k][:, i] - Xp[:, k]
					xi[k] = np.c_[xi[k], self.__f_C_MSC(Xr_k)]
		return xi

	def __f_C_MSC(self, X:np.ndarray):
		''' Converts Cartesian coordinates to MSC coordinates.

			:param X: Cartesian coordinates[x, y, z, vx, vy, vz]
			:returns: MSC coordinates
		'''
		xi = np.zeros([6, 1])
		x = X[0]
		y = X[1]
		z = X[2]
		vx = X[3]
		vy = X[4]
		vz = X[5]
		tho = np.sqrt(x**2 + y**2)
		r = np.sqrt(x**2 + y**2 + z**2)
		xi[0] = (x*vy - y*vx) / (tho*r)
		xi[1] = (vz*tho**2 - z*(x*vx + y*vy)) / (tho*r**2)
		xi[2] = (x*vx + y*vy + z*vz) / r**2
		xi[3] = np.arctan2(y, x)
		xi[4] = np.arctan(z / tho)
		xi[5] = 1 / r
		return xi

	def __GenTruth(self, model, V:str):
		''' Generates True-Track for Platforms or Targets.

			:param model: platform model of target model
			"param V: type of the model, 'platform' or 'target
			:returns: Dict contains X,t for Platforms or X,t,num,ids for Targets
		'''
		truth = {}
		if V == 'platform':
			n_x = model.X0.shape[0]										# 状态维数
			t = np.zeros([model.N_scan])
			X = np.zeros([n_x, model.N_scan])
			w = np.sqrt(model.Q) @ np.random.randn(n_x, model.N_scan)	# 过程噪声
			for k in range(model.N_scan):
				if k == 0:
					t[k] = model.t0
					X[:, [k]] = model.X0
				else:
					t[k] = (model.t0 + k * model.T) / 1000.0			# 状态对应时间
					X[:, k] = model.F @ X[:, k-1] + w[:, k-1]			# 真实状态
				# end if
			# end for
			if model.is2D:
				X[2, :] = 0
				X[5, :] = 0
			# end if
			truth['X'] = X
			truth['t'] = t
		# end if

		if V == 'target':
			n_x = model.X0.shape[0]											# 状态维数
			t = np.zeros([model.N_scan])
			num = np.zeros([1, model.N_scan])
			X = [np.zeros([6, 0]) for i in range(model.N_scan)]
			ids = [np.zeros([1, 0]).astype(np.int16) for i in range(model.N_scan)]
			for n in range(model.num):
				w = np.sqrt(model.Q) @ np.random.randn(n_x, model.N_scan)	# 过程噪声
				X_n = model.X0[:, n]
				for k in range(model.N_scan):
					if k == 0:
						t[k] = model.t0
					else:
						t[k] = (model.t0 + k * model.T) / 1000.0
					# end if
					if (t[k] >= model.lifetime[0,n]) and (t[k] <= model.lifetime[1,n]):	# 目标存在
						X_n = model.F @ X_n + w[:, k-1]
						if model.is2D:
							X_n[2] = 0
							X_n[5] = 0
						# end if
						X[k] = np.c_[X[k], X_n]
						num[0, k] += 1; 
						ids[k] = np.c_[ids[k], n]
					# end if
				# end for
			# end for
			truth['X'] = X
			truth['t'] = t
			truth['num'] = num
			truth['ids'] = ids
		# end if
		return truth
		
	def __getstate__(self):
		self_dict = self.__dict__.copy()
		del self_dict['pool']

		return self_dict

	def __setstate__(self, state):
		self.__dict__.update(state)