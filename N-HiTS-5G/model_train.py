from hyperopt import hp
from src.losses.numpy import mae, mse
from src.experiments.utils import hyperopt_tunning, model_train
from src.experiments.make_data import make_data
import os
import pickle
import time
import argparse

# Use cuda device number 0 for hp tunnig
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_experiment_space(args):
    space= {# Architecture parameters
            'model':'nhits',
            'mode': 'simple',
            'n_time_in': hp.choice('n_time_in', [3*args.horizon, 5*args.horizon, 10*args.horizon]),
            'n_time_out': hp.choice('n_time_out', [args.horizon]),
            'n_x_hidden': hp.choice('n_x_hidden', [0]),
            'n_s_hidden': hp.choice('n_s_hidden', [0]),
            'shared_weights': hp.choice('shared_weights', [False]),
            'activation': hp.choice('activation', ['ReLU', 'Tanh']),
            'initialization':  hp.choice('initialization', ['lecun_normal']),
            'stack_types': hp.choice('stack_types', [ int(args.nstacks)*['identity'] ]),
            'n_blocks': hp.choice('n_blocks', [ int(args.nstacks)*[1]]),
            'n_layers': hp.choice('n_layers', [ 9*[2] ]),
            'n_hidden': hp.choice('n_hidden', [ 512 ]),
            'n_pool_kernel_size': hp.choice('n_pool_kernel_size', [[8, 4, 1], [16, 8, 1], [1,2,4], [1,2,8]]),
            'n_freq_downsample': hp.choice('n_freq_downsample', [[168, 24, 1], [24, 12, 1], [180, 60, 1], [60, 8, 1], [40, 20, 1]]),
            'pooling_mode': hp.choice('pooling_mode', [ 'cnn', 'max', 'average']),
            'interpolation_mode': hp.choice('interpolation_mode', ['linear', 'nearest','bicubic-2']),
            # Regularization and optimization parameters
            'batch_normalization': hp.choice('batch_normalization', [True]),
            'dropout_prob_theta': hp.choice('dropout_prob_theta', [ 0 ]),
            'dropout_prob_exogenous': hp.choice('dropout_prob_exogenous', [0]),
            'learning_rate': hp.choice('learning_rate', [0.001]),
            'lr_decay': hp.choice('lr_decay', [0.5] ),
            'n_lr_decays': hp.choice('n_lr_decays', [3]), 
            'weight_decay': hp.choice('weight_decay', [0] ),
            'max_epochs': hp.choice('max_epochs', [10]),
            'max_steps': hp.choice('max_steps', [1_000]),
            'early_stop_patience': hp.choice('early_stop_patience', [10]),
            'eval_freq': hp.choice('eval_freq', [50]),
            'loss_train': hp.choice('loss', ['MAE', 'MSE']),
            'loss_hypar': hp.choice('loss_hypar', [0.5]),                
            'loss_valid': hp.choice('loss_valid', ['MAE', 'MSE']),
            'l1_theta': hp.choice('l1_theta', [0]),
            # Data parameters
            'normalizer_y': hp.choice('normalizer_y', [None]),
            'normalizer_x': hp.choice('normalizer_x', [None]),
            'complete_windows':  hp.choice('complete_windows', [True]),
            'frequency': hp.choice('frequency', ['H']),
            'seasonality': hp.choice('seasonality', [24, 60, 300]),      
            'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [10]),
            'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
            'batch_size': hp.choice('batch_size', [100]),
            'n_windows': hp.choice('n_windows', [256]),
            'random_seed': hp.quniform('random_seed', 1, 10, 1)}
    return space

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--hyperopt_max_evals', type=int, help='hyperopt_max_evals')
    parser.add_argument('--datatype', default = 'dl', required=True, type=str, help ='string to dataset type (select one in ul or dl)')
    parser.add_argument('--dataset', default=None, required=True, type=str, help='string to dataset name')
    parser.add_argument('--experiment_id', default='run1', required=False, type=str, help='string to identify experiment')
    parser.add_argument('--nstacks', default=3, required=False, type=str, help='number of stack layer')
    parser.add_argument('--exogenous', default=True, required=False, type=bool, help='use or not exogenous')
    
    return parser.parse_args()

def hp_tunning(args):        
    # make data 
    # Target Y_df
    # Exogenous X_df
    X_df, Y_df, S_df, _, _ = make_data(args.dataset, args.exogenous, make_dl = args.datatype)
    # len_val is validation dataset's length
    len_val = 7200
    # len_test is test dataset's length
    len_test = 14400
    # space is hyperparameter tunning experiment space
    space = get_experiment_space(args)
    
    # make directories to save hyperparameter tunning results 
    output_dir = f'./hp_result/{args.dataset}/{args.dataset}_{args.horizon}/'
    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'
    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'

    if not os.path.isfile(hyperopt_file):
        print('Hyperparameter optimization')
        #----------------------------------------------- Hyperopt -----------------------------------------------#
        trials = hyperopt_tunning(space=space, hyperopt_max_evals=args.hyperopt_max_evals, loss_function_val=mae,
                                  loss_functions_test={'mae':mae, 'mse': mse},
                                  Y_df=Y_df, X_df=X_df, S_df=S_df, f_cols=[],
                                  evaluate_train=True,
                                  ds_in_val=len_val, ds_in_test=len_test,
                                  return_forecasts=False,
                                  results_file = hyperopt_file,
                                  save_progress=True,
                                  loss_kwargs={})

        with open(hyperopt_file, "wb") as f:
            pickle.dump(trials, f)
    else:
        print('Hyperparameter optimization already done!')        

def get_score_min_val(dir):
    result = pickle.load(open(dir, 'rb'))
    min_mae = 100
    mc = {}
    for i in range(len(result)):
        if not('loss' in result.trials[i]['result'].keys()):
            continue
        else:
            val_mae = result.trials[i]['result']['loss']
            if val_mae < min_mae:
                mae_best = result.trials[i]['result']['test_losses']['mae']
                mse_best = result.trials[i]['result']['test_losses']['mse']
                min_mae = val_mae
                mc = result.trials[i]['result']['mc']
    return mae_best, mse_best, mc   
        
if __name__ == '__main__':
    # HP tunning
    # parse arguments
    args = parse_args()
    if args.datatype == 'dl':
        args.datatype = True
    else:
        args.datatype = False
    if args is None:
        exit()
    horizons = [10, 48, 96, 192, 336, 720]
    
    for horizon in horizons:
        print(50*'-', args.dataset, 50*'-')
        print(50*'-', horizon, 50*'-')
        start = time.time()
        args.horizon = horizon
        hp_tunning(args)
        print('Time: ', time.time() - start)
            
    # Model Train
    f_cols = []

    ds_in_val = 7200
    ds_in_test = 14400
    
    for horizon in horizons:
        dir = f'hp_result/{args.dataset}/{args.dataset}_{horizon}/hyperopt_{args.experiment_id}.p'
        _, _, mc = get_score_min_val(dir)
        mc['max_epochs'] = 2000
        
        
        X_df, Y_df, S_df, _, _ = make_data(args.dataset, args.exogenous, make_dl = args.datatype)        
        
        model_train(mc=mc, S_df=S_df, Y_df=Y_df, X_df=X_df, f_cols=f_cols, ds_in_val=ds_in_val, 
                          ds_in_test=ds_in_test, dataset_name = args.dataset, horizon = horizon, 
                          experiment_name = args.experiment_id, datatype = args.datatype)
