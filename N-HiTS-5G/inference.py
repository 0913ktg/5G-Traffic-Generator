import pickle
import torch
import argparse

from tqdm import tqdm
from src.experiments.make_data import make_data
from src.experiments.utils import *

def parse_args():
    desc = "Traffic generator"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', default=None, required=True, type=str, help='string to dataset name')
    parser.add_argument('--datatype', default = 'dl', required=True, type=str, help ='string to dataset type (select one in ul or dl)')
    parser.add_argument('--horizon', default=None, required=False, type=str, help='int to predict length')
    parser.add_argument('--experiment_id', default='run1', required=False, type=str, help='string to identify experiment')
    parser.add_argument('--size', default=None, required=False, type=str, help='int to predict volume')
    
    return parser.parse_args()

def get_score_min_val(dir):
    print(dir)
    result = pickle.load(open(dir, 'rb'))
    min_mae = 100
    mc = {}
    for i in range(len(result)-1):
        val_mae = result.trials[i]['result']['loss']
        if val_mae < min_mae:
            mae_best = result.trials[i]['result']['test_losses']['mae']
            mse_best = result.trials[i]['result']['test_losses']['mse']
            min_mae = val_mae
            mc = result.trials[i]['result']['mc']
    return mae_best, mse_best, mc

def init(dataset, horizon, data_type, experiment_id, exogenous = True):   
    if data_type == 'dl':
        make_dl = True
    else:
        make_dl = False
    _, _, _, min_max_dict, q_instance = make_data(dataset, exogenous, make_dl = make_dl)
    
    print('load train loader')
    train_loader = torch.load(f'loaders/{data_type}/{dataset}/{dataset}_{horizon}_loader.pth')  
    print('load pretrianed model')
    model = torch.load(f'models/{dataset}/{dataset}_{horizon}/{experiment_id}.pt')
    model.cuda()
    
    return model, train_loader, min_max_dict, q_instance

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
    horizons = [10, 48, 96, 192, 336, 720]
    
    for horizon in horizons:
        args.horizon = horizon
        forecast_list, _, _ = generate(args)
        
        output_dir = f'inference/{args.datatype}'
        os.makedirs(output_dir, exist_ok = True)
        assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'
        np.save(output_dir + f'/{args.dataset}_{horizon}_{args.experiment_id}.npy', forecast_list)