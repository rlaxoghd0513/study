from argparse import ArgumentParser

import yaml
from pytorch_lightning import Trainer
from gan_module import AgingGAN

parser = ArgumentParser()
parser.add_argument('--config', default='./teamproject-cycle/configs/aging_gan.yaml', help='Config to use for training')


def main():
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    model = AgingGAN(config)
    trainer = Trainer(max_epochs=config['epochs'], accelerator='gpu', auto_scale_batch_size='binsearch')
    trainer.fit(model)
    #################################추가한거#####################
    model.save_weights('./teamproject-cycle/pretrained_model/')
     ########################################
if __name__ == '__main__':
    main()

#이거 돌릴때마다 lightning_logs에 version이 하나씩 더 생김

