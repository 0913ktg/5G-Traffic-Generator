from model.GAN import GAN
import hydra
import pandas as pd
import torch

cols = ['Afreeca_DL_bitrate', 'Afreeca_UL_bitrate',
        'Amazon_DL_bitrate', 'Amazon_UL_bitrate',
        'Meet_DL_bitrate', 'Meet_UL_bitrate',
        'Navernow_DL_bitrate', 'Navernow_UL_bitrate',
        'Netflix_DL_bitrate', 'Netflix_UL_bitrate',
        'teams_DL_bitrate', 'Teams_UL_bitrate',
        'Youtube_UL_bitrate', 'Youtube_DL_bitrate',
        'YoutubeLive_DL_bitrate', 'YoutubeLive_UL_bitrate',
        'Zoom_DL_bitrate', 'Zoom_UL_bitrate',
        'TFT_DL_Bitrate', 'TFT_UL_Bitrate',
        'BattleGround_DL_Bitrate', 'BattleGround_UL_Bitrate',
        'Geforce_DL_Bitrate', 'Geforce_UL_Bitrate',
        'GameBox_DL_Bitrate', 'GameBox_UL_Bitrate',
        'Zepeto_DL_Bitrate', 'Zepeto_UL_Bitrate',
        'Roblox_DL_Bitrate', 'Roblox_UL_Bitrate']


@hydra.main(config_path='config', config_name='config')
def inference(cfg):
    version = cfg.checkpoint.version
    epoch = cfg.checkpoint.epoch

    dm = hydra.utils.instantiate(cfg.data)
    dm.setup(stage='inference')
    model = GAN.load_from_checkpoint(cfg.checkpoint.path)
    if torch.cuda.is_available():
        model = model.cuda()
    print('Load from ckpt')
    model.eval()

    print('inference start')
    df = pd.DataFrame()
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
