import os
from util.misc import super_load
from util.load_config_files import load_yaml_into_dotdict
from util.MT3DataConvertor import MT3DataConvertor
import argparse
import warnings
import pdb

os.environ['CUDA_VISIBLE_DEVICES']='2'


# Parse arguments and load the model, before doing anything else (important, reduces possibility of weird bugs)
parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', default='/home/weixinwei/study/MT3-test/src/results/2023-05-05_234310')
parser.add_argument('-tp', '--task_params', default='/home/weixinwei/study/MT3-test/src/results/2023-05-05_234310/code_used/task_params.yaml')
parser.add_argument('-mp', '--model_params', default='/home/weixinwei/study/MT3-test/src/results/2023-05-05_234310/code_used/model_params.yaml')
args = parser.parse_args()
print(f'Evaluating results from folder: {args.result_filepath}...')

# pdb.set_trace()
model, params = super_load(args.result_filepath, verbose=True)

# Test that the model was trained in the task chosen for evaluation
if args.task_params is not None:
    task_params = load_yaml_into_dotdict(args.task_params)
    for k, v in task_params.data_generation.items():
        if k not in params.data_generation:
            warnings.warn(f"Key '{k}' not found in trained model's hyperparameters")
        elif params.data_generation[k] != v:
            warnings.warn(f"Different values for key '{k}'. Task: {v}\tTrained: {params.data_generation[k]}")
    # Use task params, not the ones from the trained model
    params.recursive_update(task_params)  # note: parameters specified only on trained model will remain untouched
else:
    warnings.warn('Evaluation task was not specified; inferring it from the task specified in the results folder.')



import pickle

import numpy as np
# from data_generation.data_generator import DataGenerator
import data_generator

from modules.loss import MotLoss
from modules import evaluator

# Read evaluation hyperparameters and overwrite `params` with them
# eval_params = load_yaml_into_dotdict('configs/eval/default.yaml')
eval_params = load_yaml_into_dotdict('/home/weixinwei/study/MT3-test/configs/eval/default.yaml')
params.recursive_update(eval_params)

# data_generator = DataGenerator(params)
# GetSeqBatch = data_generator.GetSeqBatch(params)
GetSeqBatch = data_generator.GetWinBatch(params)
# mt3DataConvertor = MT3DataConvertor(args.task_params, args.model_params, evalBS = 1)
mot_loss = MotLoss(params)

og, oo, pg, d = evaluator.evaluate_metrics(GetSeqBatch, model, params, mot_loss,  num_eval=50000, verbose=False)
# og, oo, pg, d = evaluator.evaluate_metrics(mt3DataConvertor, model, params, mot_loss,  num_eval=1000, verbose=False)
# og, oo, pg, d = evaluator.evaluate_metrics(mt3DataConvertor, model, params, mot_loss,  num_eval=50000, verbose=False)

os.makedirs(os.path.join(args.result_filepath, 'eval'), exist_ok=True)
pickle.dump(og, open(os.path.join(args.result_filepath, 'eval', 'original_gospa.data'), "wb"))
pickle.dump(oo, open(os.path.join(args.result_filepath, 'eval', 'original_ospa.data'), "wb"))

print("\nFinished running evaluation... please paste this in the spread-sheet")
# Print GOSPA scores
print('GOSPA metric:')
for method, values in og.items():
    print(f"\t{method}: ")
    for value_name, value in values.items():
        print(f"\t\t{value_name:<13}: {np.mean(value):<6.4} ({np.var(value):<5.4})")

# Print OSPA scores
print('OSPA metric:')
for method, values in oo.items():
    print(f"\t{method}: ")
    for value_name, value in values.items():
        print(f"\t\t{value_name:<13}: {np.mean(value):<6.4} ({np.var(value):<5.4})")

# Print other metrics
for metric, metric_name in zip([pg, d], ['Prob-GOSPA', 'DETR']):
    print(f"{metric_name} metric:")
    for method, values in metric.items():
        print(f"\t{method:<11}: {np.mean(values):<6.4} ({np.var(values):5.4})")

# os.makedirs(os.path.join(args.result_filepath, 'eval'), exist_ok=True)
pickle.dump(og, open(os.path.join(args.result_filepath, 'eval', 'original_gospa.p'), "wb"))
pickle.dump(oo, open(os.path.join(args.result_filepath, 'eval', 'original_ospa.p'), "wb"))
pickle.dump(pg, open(os.path.join(args.result_filepath, 'eval', 'prob_gospa.p'), "wb"))
pickle.dump(d, open(os.path.join(args.result_filepath, 'eval', 'detr.p'), "wb"))
