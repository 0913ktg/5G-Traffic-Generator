from model.GAN import GAN
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import glob

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


# cols = ['Afreeca_DL_bitrate',
#        'Amazon_DL_bitrate',
#        'Meet_DL_bitrate',
#        'Navernow_DL_bitrate',
#        'Netflix_DL_bitrate',
#        'teams_DL_bitrate',
#        'Youtube_DL_bitrate',
#        'YoutubeLive_DL_bitrate',
#        'Zoom_DL_bitrate']


@hydra.main(config_path='config', config_name='config')
def my_app(cfg):
    version = 1
    epoch = 399

    dm = hydra.utils.instantiate(cfg.data)
    dm.setup(stage='inference')
    print('data module')
    model = GAN.load_from_checkpoint(f'/data/Etri/ckpts/V{version}/CLTGAN-epoch={epoch}.ckpt')
    model = model.cuda()
    print('Load from ckpt')
    model.eval()

    print('inference start')
    print(dm.cols)
    print(len(dm.conditions))
    df = pd.DataFrame()
    for i, condition in enumerate(dm.conditions):
        z = model.sample_Z(240, 300, 100)
        gen = model(z.cuda(), model.create_condition(condition, len(z)).cuda())
        gen = gen.cpu().detach().numpy().squeeze()
        gen = dm.scalers[dm.cols[i]].inverse_transform(gen.reshape(-1, 1))
        gen = gen.reshape(-1)
        # fig, ax = plt.subplots(figsize=(16, 8))
        # ax.set_title(cols[i])
        # ax.plot(gen)
        df[cols[i]] = gen
        # plt.savefig(f'/data/Etri/figures/{cols[i]}.png', bbox_inches='tight')
    df.to_csv(f'CLTGAN_V{version}_epoch_{epoch}.csv', index=False)


if __name__ == '__main__':
    my_app()
