import argparse
import datetime
import os
import pdb
import pickle
import re
import shutil
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from util.MT3DataConvertor import MT3DataConvertor
from util.TXTDataConvertor import TXTDataConvertor
from matplotlib.gridspec import GridSpec
from modules import evaluator
from modules.contrastive_loss import ContrastiveLoss
from modules.loss import FalseMeasurementLoss, MotLoss
from modules.models.mt3.mt3 import MOTT
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.load_config_files import load_yaml_into_dotdict
from util.logger import Logger
from util.SetRandomSeed import SetupSeed
from util.misc import save_checkpoint
from util.plotting import (compute_avg_certainty, get_constrastive_ax,
                           get_false_ax, get_total_loss_ax, output_truth_plot)

os.environ['CUDA_VISIBLE_DEVICES']='0'
SetupSeed(3407)
DEBUG_MODE = False

if __name__ == '__main__':
	# Load CLI arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-mp', '--model_params', help='filepath to configuration yaml file defining the model')
	parser.add_argument('--continue_training_from', help='filepath to folder of an experiment to continue training from')
	parser.add_argument('--exp_name', help='Name to give to the results folder')
	args = parser.parse_args()
	args.basePath =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	args.task_params = args.basePath  + '/configs/tasks/' + 'task1.yaml'
	args.model_params = args.basePath  + '/configs/models/' + 'mt3.pro.yaml'
	# args.continue_training_from = args.basePath + '/src/results/' + '2023-03-31_211239'
	print(f'Task configuration file: {args.task_params}')
	print(f'Model configuration file: {args.model_params}')

	# Load hyperparameters
	params = load_yaml_into_dotdict(args.task_params)
	params.update(load_yaml_into_dotdict(args.model_params))
	params.txtPathListRelative = ['Single_situation00.txt',
								  'Single_situation01.txt',
								  'Single_situation02.txt',
								  'Single_situation03.txt',
								  'Single_situation04.txt',
								  'Single_situation05.txt',
								  'Single_situation06.txt',
								  'Single_situation07.txt',
								  'Single_situation08.txt',
								  'Single_situation09.txt',
								  'Single_situation10.txt',
								  'Single_situation11.txt',
								  'Single_situation12.txt',
								  'Single_situation13.txt',
								  'Single_situation14.txt',
								  'Single_situation15.txt',
								  'Single_situation16.txt',
								  'Single_situation17.txt',
								  'Single_situation18.txt',
								  'Single_situation19.txt']
	params.txtPathList = [args.basePath + '/source/' + i for i in params.txtPathListRelative]
	assert len(params.txtPathList) == params.training.batch_size, f'The number of txt files({len(params.txtPathList)}) should be equal to batch size({params.training.batch_size}).'

	if params.training.device == 'auto':
		params.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	if not DEBUG_MODE:
		# Create logger and save all code dependencies imported so far
		cur_path = os.path.dirname(os.path.abspath(__file__))
		results_folder_path = cur_path + os.sep + 'results'
		exp_name = args.exp_name if args.exp_name is not None else time.strftime("%Y-%m-%d_%H%M%S")
		
		logger = Logger(log_path=f'{results_folder_path}/{exp_name}', save_output=False)
		print(f"Saving results to folder {logger.log_path}")

		# -------------------恢复------------------------
		logger.save_code_dependencies(project_root_path=os.path.realpath('../'))  # assuming this is ran from repo root

		# Manually copy the configuration yaml file used for this experiment to the logger folder
		shutil.copy(args.task_params, os.path.join(logger.log_path, 'code_used', 'task_params.yaml'))
		shutil.copy(args.model_params, os.path.join(logger.log_path, 'code_used', 'model_params.yaml'))

		# If continuing an experiment, manually copy the `code_used` of the experiment from which training wil continue
		if args.continue_training_from is not None:
			try:
				shutil.copytree(os.path.join(args.continue_training_from, 'code_used'),
								os.path.join(logger.log_path, 'code_from_previous_training'))
			except FileNotFoundError:
				print(f'Path specified to continue training from does not exist: {args.continue_training_from}')
				exit()
		# ------------------------------------------------

	model = MOTT(params)
	mt3DataConvertor = MT3DataConvertor(params.txtPathList,
				    					params.data_generation.n_timesteps * params.data_generation.frame_sample_rate,
										params.training.batch_size,
										params.data_generation.frame_sample_rate,
										params.training.device)
	# txtDataConvertor = TXTDataConvertor(params.txtPathList, params.data_generation.n_timesteps, params.training.batch_size)
	mot_loss = MotLoss(params)
	contrastive_loss = ContrastiveLoss(params)
	false_loss = FalseMeasurementLoss(params)

	# plt.figure(figsize=(8, 8), dpi=300)
	# for i in range(100):
	# 	txtDataConvertor.plotTrain()

	# Optionally load the model weights from a provided checkpoint
	if args.continue_training_from is not None:
		# Find filename for last checkpoint available
		checkpoints_path = os.path.join(args.continue_training_from, 'checkpoints')
		checkpoint_names = os.listdir(checkpoints_path)
		idx_last = np.argmax([int(re.findall(r"\d+", c)[-1]) for c in checkpoint_names])  # extract last occurrence of a number from the names
		last_filename = os.path.join(checkpoints_path, checkpoint_names[idx_last])

		# Load model weights and pass model to correct device
		checkpoint = torch.load(last_filename)
		model.load_state_dict(checkpoint['model_state_dict'])

	model.to(torch.device(params.training.device))
	optimizer = Adam(model.parameters(), lr=params.training.learning_rate)
	scheduler = ReduceLROnPlateau(optimizer,
								  patience=params.training.reduce_lr_patience,
								  factor=params.training.reduce_lr_factor,
								  verbose=params.debug.print_reduce_lr_messages)
	# Optionally load optimizer and scheduler states from provided checkpoint (this has to be done after loading the
	# model weights and calling model.to(), to guarantee these will be in the correct device too)
	if args.continue_training_from is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

	current_lr = optimizer.param_groups[0]['lr']
	if not DEBUG_MODE:
		logger.log_scalar('metrics/learning_rate', current_lr, 0)

		if params.debug.enable_plot or params.debug.save_plot_figs:
			fig = plt.figure(constrained_layout=True, figsize=(15, 8))
			fig.canvas.set_window_title('Training Progress')

			gs = GridSpec(2, 3, figure=fig)
			loss_ax = fig.add_subplot(gs[0, 0])
			loss_ax.set_ylabel('Loss', color='C0')
			loss_ax.grid('on')
			loss_line, = loss_ax.plot([1], 'r', label='Loss', c='C0')
			loss_ax.tick_params(axis='y', labelcolor='C0')
			loss_ax.set_yscale('log')

			percent_ax = fig.add_subplot(gs[1, 0])
			percent_ax.set_ylabel('Certainty distribution')
			percent_ax.grid('on')
			matched_median_cert_line, = percent_ax.plot([1], 'C0', label='Matched median certainty')
			unmatched_median_cert_line, = percent_ax.plot([1], 'C3', label='Unmatched median certainty')
			max_cert_line, = percent_ax.plot([1], 'C0--', label='Max certainty')
			min_cert_line, = percent_ax.plot([1], 'C0--', label='Min certainty')
			
			output_ax = fig.add_subplot(gs[:, 1:])
			output_ax.set_ylabel('Y')
			output_ax.set_xlabel('X')

			if params.debug.save_plot_figs:
				os.makedirs(os.path.join(logger.log_path, 'figs', 'main'))
				total_loss_fig, total_loss_ax, total_loss_line = get_total_loss_ax()
				os.makedirs(os.path.join(logger.log_path, 'figs', 'aux'))

			if params.loss.contrastive_classifier:
				contrastive_loss_fig, contrastive_loss_ax, contrastive_loss_line = get_constrastive_ax()
				os.makedirs(os.path.join(logger.log_path, 'figs', 'aux', 'contrastive'))
				
			if params.loss.false_classifier:
				false_loss_fig, false_loss_ax, false_loss_line = get_false_ax()
				os.makedirs(os.path.join(logger.log_path, 'figs', 'aux', 'false'))
	# end if

	losses = []
	last_layer_losses = []
	c_losses = []
	f_losses = []
	matched_min_certainties = []
	matched_q1_certainties = []
	matched_median_certainties = []
	matched_q3_certainties = []
	matched_max_certainties = []
	unmatched_min_certainties = []
	unmatched_q1_certainties = []
	unmatched_median_certainties = []
	unmatched_q3_certainties = []
	unmatched_max_certainties = []

	outputs_history = deque(maxlen=50)
	indices_history = deque(maxlen=50)

	print("[INFO] Training started...")
	start_time = time.time()
	time_since = time.time()

	for i_gradient_step in range(params.training.n_gradient_steps):
		try:
			batch, panValue, labels, unique_ids = mt3DataConvertor.Get_batch()
			outputs, memory, aux_classifications, queries, attn_maps  = model.forward(batch, panValue, unique_ids)
			loss_dict, indices = mot_loss.forward(outputs, labels, loss_type=params.loss.type)

			if params.loss.type == 'both':
				gospa_weight = (i_gradient_step / params.training.n_gradient_steps)**3
				total_loss = sum(gospa_weight * loss_dict[k] if ('gospa' in k) else loss_dict[k] for k in loss_dict.keys())
				if not DEBUG_MODE:
					logger.log_scalar(f'metrics/detr', loss_dict['detr'], i_gradient_step)
					logger.log_scalar(f'metrics/gospa', loss_dict['gospa'], i_gradient_step)
				last_layer_losses.append(loss_dict['detr'].item() + loss_dict['gospa'].item()*gospa_weight)
			else:
				total_loss = sum(loss_dict[k] for k in loss_dict.keys())
				last_layer_losses.append(loss_dict[params.loss.type].item())
				if not DEBUG_MODE:
					logger.log_scalar(f'metrics/{params.loss.type}', loss_dict[params.loss.type], i_gradient_step)

			if params.loss.contrastive_classifier:
				c_loss = contrastive_loss(aux_classifications['contrastive_classifications'], unique_ids)
				total_loss = total_loss + c_loss * params.loss.c_loss_multiplier
				c_losses.append(c_loss.item())
				if not DEBUG_MODE:
					logger.log_scalar('metrics/contrastive_loss', c_loss, i_gradient_step)

			if params.loss.false_classifier:
				f_loss = false_loss(aux_classifications['false_classifications'], unique_ids)
				total_loss = total_loss + f_loss * params.loss.f_loss_multiplier
				f_losses.append(f_loss.item())
				if not DEBUG_MODE:
					logger.log_scalar('metrics/false_loss', f_loss, i_gradient_step)

			losses.append(total_loss.item())

			# Compute quantiles for matched and unmatched predictions
			outputs_history.append({'state': outputs['state'].detach().cpu(), 'logits': outputs['logits'].detach().cpu()})
			indices_history.append(indices)
			matched_quants, unmatched_quants = compute_avg_certainty(outputs_history, indices_history)
			min_cert, q1_cert, median_cert, q3_cert, max_cert = matched_quants
			matched_min_certainties.append(min_cert)
			matched_q1_certainties.append(q1_cert)
			matched_median_certainties.append(median_cert)
			matched_q3_certainties.append(q3_cert)
			matched_max_certainties.append(max_cert)
			
			min_cert, q1_cert, median_cert, q3_cert, max_cert = unmatched_quants
			unmatched_min_certainties.append(min_cert)
			unmatched_q1_certainties.append(q1_cert)
			unmatched_median_certainties.append(median_cert)
			unmatched_q3_certainties.append(q3_cert)
			unmatched_max_certainties.append(max_cert)

			if not DEBUG_MODE:
				logger.log_scalar('metrics/total_loss', total_loss.item(), i_gradient_step)
				if params.loss.return_intermediate:
					for k, v in loss_dict.items():
						if '_' in k:
							logger.log_scalar('metrics/'+k, v.item(), i_gradient_step)
				logger.log_scalar('metrics/matched_min_certainty', min_cert, i_gradient_step)
				logger.log_scalar('metrics/matched_q1_certainty', q1_cert, i_gradient_step)
				logger.log_scalar('metrics/matched_median_certainty', median_cert, i_gradient_step)
				logger.log_scalar('metrics/matched_q3_certainty', q3_cert, i_gradient_step)
				logger.log_scalar('metrics/matched_max_certainty', max_cert, i_gradient_step)
				logger.log_scalar('metrics/unmatched_min_certainty', min_cert, i_gradient_step)
				logger.log_scalar('metrics/unmatched_q1_certainty', q1_cert, i_gradient_step)
				logger.log_scalar('metrics/unmatched_median_certainty', median_cert, i_gradient_step)
				logger.log_scalar('metrics/unmatched_q3_certainty', q3_cert, i_gradient_step)
				logger.log_scalar('metrics/unmatched_max_certainty', max_cert, i_gradient_step)
			
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			# Update learning rate, logging it if changed
			scheduler.step(total_loss)
			new_lr = optimizer.param_groups[0]['lr']
			if new_lr != current_lr:
				current_lr = new_lr
				if not DEBUG_MODE:
					logger.log_scalar('metrics/learning_rate', current_lr, i_gradient_step)

			if i_gradient_step % params.debug.print_interval == 0:
				cur_time = time.time()
				t = str(datetime.timedelta(seconds=round(cur_time - time_since)))
				t_tot = str(datetime.timedelta(seconds=round(cur_time - start_time)))
				print(f"Number of gradient steps: {i_gradient_step + 1} \t "
					  f"Loss: {np.mean(losses[-15:])} \t "
					  f"Time per step: {(cur_time-time_since)/params.debug.print_interval} \t "
					  f"Total time elapsed: {t_tot}")
				time_since = time.time()
			
			if not DEBUG_MODE:
				if (params.debug.enable_plot and i_gradient_step % params.debug.plot_interval == 0) or \
						(params.debug.save_plot_figs and i_gradient_step % params.debug.save_plot_figs_interval == 0):
					x_axis = list(range(i_gradient_step+1))
					loss_line.set_data(x_axis, last_layer_losses)
					loss_ax.relim()
					loss_ax.autoscale_view()

					percent_ax.collections.clear()
					matched_median_cert_line.set_data(x_axis, np.array(matched_median_certainties))
					percent_ax.fill_between(x_axis, matched_min_certainties, matched_max_certainties, color='C0', alpha=0.3, linewidth=0.0)
					percent_ax.fill_between(x_axis, matched_q1_certainties, matched_q3_certainties, color='C0', alpha=0.6, linewidth=0.0)
					unmatched_median_cert_line.set_data(x_axis, np.array(unmatched_median_certainties))
					percent_ax.fill_between(x_axis, unmatched_min_certainties, unmatched_max_certainties, color='C3', alpha=0.3, linewidth=0.0)
					percent_ax.fill_between(x_axis, unmatched_q1_certainties, unmatched_q3_certainties, color='C3', alpha=0.6, linewidth=0.0)
					percent_ax.set_ylim([-0.05, 1.05])

					output_ax.cla()
					output_ax.grid('on')
					output_truth_plot(output_ax, outputs, labels, indices, batch)
					output_ax.set_xlim([params.data_generation.field_of_view_lb, params.data_generation.field_of_view_ub])
					output_ax.set_ylim([params.data_generation.field_of_view_lb, params.data_generation.field_of_view_ub])

					if params.loss.contrastive_classifier:
						contrastive_loss_line.set_data(x_axis, c_losses)
						contrastive_loss_ax.relim()
						contrastive_loss_ax.autoscale_view()

					if params.loss.false_classifier:
						false_loss_line.set_data(x_axis, f_losses)
						false_loss_ax.relim()
						false_loss_ax.autoscale_view()

					if (params.debug.enable_plot and i_gradient_step % params.debug.plot_interval == 0):
						fig.canvas.draw()
						plt.pause(0.01)

					if params.debug.save_plot_figs and i_gradient_step % params.debug.save_plot_figs_interval == 0:
						filename = f"gradient_step{i_gradient_step}.jpg"
						fig.savefig(os.path.join(logger.log_path, 'figs', 'main', filename))

						total_loss_line.set_data(x_axis, losses)
						total_loss_ax.relim()
						total_loss_ax.autoscale_view()
						total_loss_fig.savefig(os.path.join(logger.log_path, 'figs', 'aux', filename))

						if params.loss.contrastive_classifier:
							contrastive_loss_fig.savefig(os.path.join(logger.log_path, 'figs', 'aux', 'contrastive', filename))
						if params.loss.false_classifier:
							false_loss_fig.savefig(os.path.join(logger.log_path, 'figs', 'aux', 'false', filename))

		except KeyboardInterrupt:
			if not DEBUG_MODE:
				filename = f'checkpoint_gradient_step_{i_gradient_step}'
				folder_name = os.path.join(logger.log_path, 'checkpoints')
				save_checkpoint(folder=folder_name,
								filename=filename,
								model=model,
								optimizer=optimizer,
								scheduler=scheduler)
			print("[INFO] Exiting...")
			exit()

		# Save checkpoint
		if (i_gradient_step+1) % params.training.checkpoint_interval == 0:
			if not DEBUG_MODE:
				filename = f'checkpoint_gradient_step_{i_gradient_step}'
				folder_name = os.path.join(logger.log_path, 'checkpoints')
				save_checkpoint(folder=folder_name,
								filename=filename,
								model=model,
								optimizer=optimizer,
								scheduler=scheduler)

	print("[INFO] Training finished.")