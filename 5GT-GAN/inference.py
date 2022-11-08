from model.GAN import GAN
import hydra
import pandas as pd
import torch


@hydra.main(config_path='config', config_name='config')
def inference(cfg):
    version = cfg.checkpoint.version
    epoch = cfg.checkpoint.epoch

    # Get columns for create conditions
    df = pd.read_csv(cfg.data.data_path)
    cols = df.columns[~df.columns.str.contains('Unnamed')]

    # Get datamodule for create conditions and fit scalers
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup(stage='inference')
    print('Load from ckpt')
    model = GAN.load_from_checkpoint(cfg.checkpoint.path)
    if torch.cuda.is_available():
        model = model.cuda()
    print('Model loaded')
    model.eval()

    print('inference start')
    df = pd.DataFrame()
    # Generate data for each conditions
    for i, condition in enumerate(dm.conditions):
        z = model.sample_Z(cfg.checkpoint.batch_size, 300, 100)
        gen = model(z.cuda(), model.create_condition(condition, len(z)).cuda())
        gen = gen.cpu().detach().numpy().squeeze()
        gen = dm.scalers[dm.cols[i]].inverse_transform(gen.reshape(-1, 1))
        gen = gen.reshape(-1)
        df[cols[i]] = gen
    df.to_csv(f'5GTGAN_V{version}_epoch_{epoch}.csv', index=False)


if __name__ == '__main__':
    inference()
