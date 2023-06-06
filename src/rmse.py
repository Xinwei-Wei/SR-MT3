import argparse
import os
import warnings

from util.MT3DataConvertor import MT3DataConvertor
import PerformanceEval
from util.load_config_files import load_yaml_into_dotdict
from util.misc import super_load
import scipy.io as scio

os.environ['CUDA_VISIBLE_DEVICES']='1'

# Parse arguments and load the model, before doing anything else (important, reduces possibility of weird bugs)
scene = '2.5'
mat = '2_5'
result = '2023-03-31_211406'
parser = argparse.ArgumentParser()
parser.add_argument('-rp', '--result_filepath', default=f'/home/weixinwei/study/MT3-test/src/results/{result}')
parser.add_argument('-tp', '--task_params', default=f'/home/weixinwei/study/MT3-test/configs/tasks/scen{scene}.yaml')
parser.add_argument('-mp', '--model_params', default='/home/weixinwei/study/MT3-test/configs/models/mt3.pro.yaml')
args = parser.parse_known_args()[0]
print(f'Evaluating results from folder: {args.result_filepath}...')
print(f'Scene: {scene}')

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

eval_params = load_yaml_into_dotdict('/home/weixinwei/study/MT3-test/configs/eval/default.yaml')
params.recursive_update(eval_params)
mt3DataConvertor = MT3DataConvertor(args.task_params, args.model_params, 1)

xRMSE, yRMSE = PerformanceEval.CalculateRMSE(mt3DataConvertor, model, 71, 1, 20000, 10, 0.4)

RMSE = {f'RMSE{mat}':{'scene':f'{mat}', 'xRMSE': xRMSE, 'yRMSE': yRMSE}}
scio.savemat(f'/home/weixinwei/toMATLAB90/RMSE{mat}.mat', RMSE)
print(f'MAT saved to RMSE_{mat}.mat')