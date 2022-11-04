import pickle
import torch
import argparse
from tqdm import tqdm
from src.experiments.make_data import make_data
from src.experiments.utils import *

# Use cuda device number 0 for hp tunnig
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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

def parse_args():
    desc = "Traffic generator"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', default=None, required=True, type=str, help='string to dataset name')
    parser.add_argument('--datatype', default = 'dl', required=True, type=str, help ='string to dataset type (select one in ul or dl)')
    parser.add_argument('--horizon', default=None, required=False, type=str, help='int to predict length')
    parser.add_argument('--experiment_id', default='run1', required=False, type=str, help='string to identify experiment')
    parser.add_argument('--size', default=None, required=False, type=str, help='int to predict volume')
    
    return parser.parse_args()

def init(dataset, horizon, data_type, experiment_id, exogenous = True):   
    
    # create model and load checkpoint
    dir = f'hp_result/{dataset}/{dataset}_{horizon}/hyperopt_{data_type}.p'
    _, _, mc = get_score_min_val(dir)
    
    model = instantiate_model(mc=mc)
    ckpt = torch.load(f'best_ckpt/{dataset}/{dataset}_{horizon}/{data_type}.ckpt')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)    
    
    make_dl = True if data_type == 'dl' else False
    X_df, Y_df, S_df, min_max_dict, q_instance = make_data(dataset, exogenous, make_dl = make_dl)
    
    # make datasets
    f_cols, ds_in_val, ds_in_test = [], 7200, 14400
    train_dataset, val_dataset, test_dataset, _ = create_datasets(mc=mc,
                                                                S_df=S_df, Y_df=Y_df, X_df=X_df,
                                                                f_cols=f_cols,
                                                                ds_in_val=ds_in_val,
                                                                ds_in_test=ds_in_test)
    mc['n_x'], mc['n_s'] = train_dataset.get_n_variables()
    
    # make loader
    loader, _, _ = instantiate_loaders(mc=mc,
                                        train_dataset=train_dataset,
                                        val_dataset=val_dataset,
                                        test_dataset=test_dataset) 
    return model, loader, min_max_dict, q_instance

def generate(args):    
    
    if args.size == None:
        args.size = 72000//horizon
    
    model, train_loader, min_max_dict, q_instance = init(dataset = args.dataset, horizon = args.horizon, 
                                                         data_type=args.datatype, experiment_id = args.experiment_id)   
    
    forecast_list = []

    for i in tqdm(range(int(args.size))):
        batch = next(iter(train_loader))
        # 학습 안되게 변경 해야함
        loss, outsample_y, forecast = model.training_step(batch, is_inf=True)   
        forecast_list.extend(list(forecast[0].cpu().detach().numpy().flatten()))

    forecast_list = [-1 if x < -1 else x for x in forecast_list]
    print(min_max_dict.keys())
    max, min = min_max_dict[f'{args.datatype.upper()}_bitrate'][0], min_max_dict[f'{args.datatype.upper()}_bitrate'][1]    
    
    print('denormalizing traffic')
    forecast_list = denormalize(np.array(forecast_list), max, min)
    
    q = q_instance[f'{args.datatype.upper()}_bitrate']
    forecast_list = forecast_list.reshape(-1,1)
    forecast_list = q.inverse_transform(forecast_list)
    forecast_list = forecast_list.flatten()
    
    return forecast_list, outsample_y, forecast

if __name__ == "__main__":    
    args = parse_args()    
    datasets = ['afreeca','amazon','meet','navernow','netflix','teams'
                ,'youtube','youtubeLive','zoom','battleground','gamebox','geforce'
                ,'roblox','tft','zepeto']
    # horizons = [10, 48, 96, 192, 336, 720]
    horizons = [10]
    
    for horizon in horizons:
        args.horizon = horizon
        forecast_list, _, _ = generate(args)
        
        output_dir = f'inference/{args.datatype}'
        os.makedirs(output_dir, exist_ok = True)
        assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'
        np.save(output_dir + f'/{args.dataset}_{horizon}_{args.experiment_id}.npy', forecast_list)