import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
from src.evaluation.evaluation import *

# set your dataset and horizon
datasets = ['afreeca','amazon','meet','navernow','netflix','teams'
                ,'youtube','youtubeLive','zoom','battleground','gamebox','geforce'
                ,'roblox','tft','zepeto']
horizons = [10, 48, 96, 192, 336, 720]

datatype = 'dl'
experiment_id = 'test1'

# evaluate use jsd and mmd^2
jsdmmd2_dict = {}

for dataset in datasets:
    horizon_mmd_scores = []
    horizon_jsd_scores = []
    for horizon in horizons:
        traffic = np.load(f'inference/{datatype}/{dataset}_{horizon}_{experiment_id}.npy')
        real = pd.read_csv(f'../data/{dataset}_dataset.csv')
        real = real.dropna()
        
        if len(real) > len(traffic):
            data_len = len(traffic)
        else:
            data_len = len(real)   
        
        mmd_score = rbf_mmd2(np.array(real[f'{datatype.upper()}_bitrate'][:data_len]), traffic[:data_len])
        jsd_score = jsd(real[f'{datatype.upper()}_bitrate'][:data_len].to_numpy(), traffic[:data_len])        
        
        horizon_jsd_scores.append(round(jsd_score, 5))
        horizon_mmd_scores.append(round(mmd_score, 5))
        
    jsdmmd2_dict[dataset] = [horizon_jsd_scores, horizon_mmd_scores]
    
output_dir = f'evaluation/jsd&mmd2'
os.makedirs(output_dir, exist_ok = True)
assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

## save dictionary
with open(output_dir + f'/{experiment_id}.pickle','wb') as fw:
    pickle.dump(jsdmmd2_dict, fw)
    
