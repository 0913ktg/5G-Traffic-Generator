import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from src.experiments.utils import normalize

def data_preprocess(df):    
    min_max_dict = {}
    q_instance = {}

    for column in df.columns:
        # quantile norm
        qt = QuantileTransformer(output_distribution = 'normal')
        temp = df[column].to_numpy().reshape(-1, 1)
        temp = qt.fit_transform(temp)
        q_instance[column] = qt

        df[column] = temp.flatten()
        df[column], min_max = normalize(df[column])
        min_max_dict[column] = min_max

    all_data = []
    unique_id_list = []
    cnt = 65

    for data in df.values:
        all_data.extend(data)
        unique_id_list.extend([chr(x) for x in range(cnt, cnt+len(data))])

    ds = pd.date_range(start="2018-4-1", periods=72000, freq='T')
    time = ds.array.to_numpy().tolist()
    new_time = []

    for t in time:
        for _ in range(len(df.columns)):
            new_time.append(t)

    ds = pd.DatetimeIndex(new_time)
    df = pd.DataFrame([list(x) for x in zip(ds, unique_id_list, all_data)], columns=['ds', 'unique_id', 'y'])
    
    return df, min_max_dict, q_instance

def make_data(dataset, exogenous = True, make_dl = True):
    df = pd.read_csv(f'../data/{dataset}_dataset.csv')
    # 72000개 단위로 데이터 제한
    df.drop(index = [x for x in range(72000, len(df))], inplace = True)
            
    if exogenous:
        print(f'load_{dataset}_X_df dataset')
        if make_dl:
            X_df = df['UL_bitrate'].to_frame()
        else:
            X_df = df['DL_bitrate'].to_frame()
        print('X_df make_dataset')
        X_df, _, _ = data_preprocess(X_df)
        X_df.columns = ['ds', 'unique_id', 'x']
        
    print(f'load_{dataset}_Y_df dataset')
    if make_dl:
        Y_df = df['DL_bitrate'].to_frame()
    else:
        Y_df = df['UL_bitrate'].to_frame()
    
    print('Y_df make_dataset')
    Y_df, min_max_dict, q_instance = data_preprocess(Y_df)   
    
    S_df = None
    
    return X_df, Y_df, S_df, min_max_dict, q_instance