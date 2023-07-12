import itertools
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from dataset import ImagetoImageDataset
from models import Generator, Discriminator


class AgingGAN(pl.LightningModule):

    def __init__(self, config):
        super(AgingGAN, self).__init__()
        self.config = config
        self.genA2B = Generator(config['ngf'], n_residual_blocks=config['n_blocks'])
        self.genB2A = Generator(config['ngf'], n_residual_blocks=config['n_blocks'])
        self.disGA = Discriminator(config['ndf'])
        self.disGB = Discriminator(config['ndf'])

        # cache for generated images
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

    def forward(self, x):
        return self.genA2B(x) #에이징
        # return self.genB2A(x) #디에이징

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        if optimizer_idx == 0:
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = self.genA2B(real_B)
            # same_B = self.genB2A(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B) * self.config['identity_weight']
            # G_B2A(A) should equal A if real A is fed
            same_A = self.genB2A(real_A)
            # same_A = self.genA2B(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * self.config['identity_weight']

            # GAN loss
            fake_B = self.genA2B(real_A)
            # fake_B = self.genB2A(real_A)
            pred_fake = self.disGB(fake_B)
            loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.config[
                'adv_weight']

            fake_A = self.genB2A(real_B)
            # fake_A = self.genA2B(real_B)
            pred_fake = self.disGA(fake_A)
            loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones(pred_fake.shape).type_as(pred_fake)) * self.config[
                'adv_weight']

            # Cycle loss
            recovered_A = self.genB2A(fake_B)
            # recovered_A = self.genA2B(fake_B)
            loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.config['cycle_weight']

            recovered_B = self.genA2B(fake_A)
            # recovered_B = self.genB2A(fake_A)
            loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.config['cycle_weight']

            # Total loss
            g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            output = {
                'loss': g_loss,
                'log': {'Loss/Generator': g_loss}
            }
            self.generated_B = fake_B
            self.generated_A = fake_A

            self.real_B = real_B
            self.real_A = real_A

            # Log to tb
            if batch_idx % 500 == 0:
                self.logger.experiment.add_image('Real/A', make_grid(self.real_A, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Real/B', make_grid(self.real_B, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Generated/A',
                                                 make_grid(self.generated_A, normalize=True, scale_each=True),
                                                 self.current_epoch)
                self.logger.experiment.add_image('Generated/B',
                                                 make_grid(self.generated_B, normalize=True, scale_each=True),
                                                 self.current_epoch)
            return output

        if optimizer_idx == 1:
            # Real loss
            pred_real = self.disGA(real_A)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            # Fake loss
            fake_A = self.generated_A
            pred_fake = self.disGA(fake_A.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # Real loss
            pred_real = self.disGB(real_B)
            loss_D_real = F.mse_loss(pred_real, torch.ones(pred_real.shape).type_as(pred_real))

            # Fake loss
            fake_B = self.generated_B
            pred_fake = self.disGB(fake_B.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros(pred_fake.shape).type_as(pred_fake))

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            d_loss = loss_D_A + loss_D_B
            output = {
                'loss': d_loss,
                'log': {'Loss/Discriminator': d_loss}
            }
            return output

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                   lr=self.config['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.config['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(),
                                                   self.disGB.parameters()),
                                   lr=self.config['lr'],
                                   betas=(0.5, 0.999),
                                   weight_decay=self.config['weight_decay'])
        return [g_optim, d_optim], []

    def train_dataloader(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.config['img_size'] + 30, self.config['img_size'] + 30)),
            transforms.RandomCrop(self.config['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = ImagetoImageDataset(self.config['domainA_dir'], self.config['domainB_dir'], train_transform)
        return DataLoader(dataset,
                          batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'],
                          shuffle=True)

##########추가한거###################
    # def save_weights(self, path):
    #     name = '1.pth'
    #     full_path = os.path.join(path, name)
    #     torch.save(self.state_dict(), full_path)

    def save_weights(self, path):
        generator_state_dict = self.genA2B.state_dict()
        discriminator_state_dict = self.disGB.state_dict()
        state_dict = {
            'genA2B': generator_state_dict,
            'disGB': discriminator_state_dict
        }
        torch.save(state_dict, os.path.join('./teamproject-cycle/pretrained_model/state_dict_korea_aging.pth'))

    def load_weights(self, path):
        checkpoint = torch.load(os.path.join('./teamproject-cycle/pretrained_model/state_dict_korea_aging.pth'))
        generator_state_dict = checkpoint['genA2B']
        discriminator_state_dict = checkpoint['disGB']
        self.genA2B.load_state_dict(generator_state_dict)
        self.disGB.load_state_dict(discriminator_state_dict)