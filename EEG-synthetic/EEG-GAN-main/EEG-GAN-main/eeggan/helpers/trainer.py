from doctest import debug_script
import os
import time
from tqdm import tqdm
from decimal import Decimal
import numpy as np
import torch.nn as nn
import torch
import copy
from torch.utils.data import DataLoader
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

import eeggan.nn_architecture.losses as losses
from eeggan.nn_architecture.losses import WassersteinGradientPenaltyLoss as Loss
from eeggan.nn_architecture.models import DecoderGenerator, EncoderDiscriminator


class Trainer:
    def __init__(self):
        pass

    def training(self):
        raise NotImplementedError

    def batch_train(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

    def manage_checkpoints(self):
        raise NotImplementedError

    def print_log(self):
        raise NotImplementedError


class GANTrainer(Trainer):
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt):
        # training configuration
        super().__init__()
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = opt['sequence_length'] if 'sequence_length' in opt else 0
        self.input_sequence_length = opt['input_sequence_length'] if 'input_sequence_length' in opt else 0
        self.sequence_length_generated = self.sequence_length
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.latent_dim = opt['latent_dim'] if 'latent_dim' in opt else 10
        self.critic_iterations = opt['critic_iterations'] if 'critic_iterations' in opt else 5
        self.lambda_gp = opt['lambda_gp'] if 'lambda_gp' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.d_lr = opt['discriminator_lr'] if 'learning_rate' in opt else 0.0001
        self.g_lr = opt['generator_lr'] if 'learning_rate' in opt else 0.0001
        self.n_conditions = opt['n_conditions'] if 'n_conditions' in opt else 0
        self.n_channels = opt['n_channels'] if 'n_channels' in opt else 1
        self.channel_names = opt['channel_names'] if 'channel_names' in opt else list(range(0, self.n_channels))
        self.b1, self.b2 = 0, 0.9  # alternative values: .5, 0.999
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank
        self.start_time = time.time()

        self.generator = generator
        self.discriminator = discriminator
        if hasattr(generator, 'module'):
            self.generator = {k.partition('module.')[2]: v for k,v in generator}
            self.discriminator = {k.partition('module.')[2]: v for k,v in discriminator}

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr=self.g_lr, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                lr=self.d_lr, betas=(self.b1, self.b2))

        self.loss = Loss()
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            self.loss.set_lambda_gp(self.lambda_gp)

        self.d_losses = []
        self.g_losses = []
        self.trained_epochs = 0

        self.prev_g_loss = 0
        generator_class = str(self.generator.__class__.__name__) if not isinstance(self.generator, DecoderGenerator) else str(self.generator.generator.__class__.__name__)
        discriminator_class = str(self.discriminator.__class__.__name__) if not isinstance(self.discriminator, EncoderDiscriminator) else str(self.discriminator.discriminator.__class__.__name__)
        self.configuration = {
            'device': self.device,
            'generator_class': generator_class,
            'discriminator_class': discriminator_class,
            'model_class': 'GAN',
            'sequence_length': self.sequence_length,
            'sequence_length_generated': self.sequence_length_generated,
            'num_layers': opt['num_layers'],
            'hidden_dim': opt['hidden_dim'],
            'latent_dim': self.latent_dim,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'trained_epochs': self.trained_epochs,
            'sample_interval': self.sample_interval,
            'discriminator_lr': self.d_lr,
            'generator_lr': self.g_lr,
            'n_conditions': self.n_conditions,
            'latent_dim': self.latent_dim,
            'critic_iterations': self.critic_iterations,
            'lambda_gp': self.lambda_gp,
            'patch_size': opt['patch_size'] if 'patch_size' in opt else None,
            'b1': self.b1,
            'b2': self.b2,
            'data': opt['data'] if 'data' in opt else None,
            'autoencoder': opt['autoencoder'] if 'autoencoder' in opt else None,
            'n_channels': self.n_channels,
            'channel_names': self.channel_names,
            'seed': opt['seed'],
            'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
            'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
            'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            'save_name': opt['save_name'] if 'save_name' in opt else '',
            'dataloader': {
                'data': opt['data'] if 'data' in opt else None,
                'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
                'diff_data': opt['diff_data'] if 'diff_data' in opt else None,
                'std_data': opt['std_data'] if 'std_data' in opt else None,
                'norm_data': opt['norm_data'] if 'norm_data' in opt else None,
                'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
                'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            },
            'history': opt['history'] if 'history' in opt else {},
        }
        # Dentro GANTrainer.__init__
        if isinstance(self.discriminator, EncoderDiscriminator):
            base_disc = self.discriminator.discriminator
        else:
            base_disc = self.discriminator

        # Questa è la parte del discriminatore che produce il vettore z da 128
        self.latent_encoder = nn.Sequential(
            copy.deepcopy(base_disc[0]), # Patch Embedding
            copy.deepcopy(base_disc[1]), # Transformer Blocks
            copy.deepcopy(base_disc[2].clshead[0]), # Reduce
            copy.deepcopy(base_disc[2].clshead[1])  # LayerNorm
        ).to(self.device)
        for param in self.latent_encoder.parameters():
            param.requires_grad = False
        self.latent_encoder.eval()

        # Definiamo l'intensità della perturbazione (Epsilon)
        self.epsilon = 0.05 # Puoi variarlo tra 0.01 e 0.1
    def training(self, dataset: DataLoader):
        """Batch training of the conditional Wasserstein-GAN with GP."""
        gen_samples = []
        # checkpoint file settings; toggle between two checkpoints to avoid corrupted file if training is interrupted
        path_checkpoint = 'trained_models'
        trigger_checkpoint_01 = True
        checkpoint_01_file = 'checkpoint_01.pt'
        checkpoint_02_file = 'checkpoint_02.pt'

        gen_samples_batch = None
        batch = None

        loop = tqdm(range(self.epochs))
        # try/except for KeyboardInterrupt --> Abort training and save model
        try:
            for epoch in loop:
                # for-loop for number of batch_size entries in sessions
                i_batch = 0
                d_loss_batch = 0
                g_loss_batch = 0
                for batch in dataset:
                    # draw batch_size samples from sessions
                    data = batch[:, self.n_conditions:].to(self.device)
                    data_labels = batch[:, :self.n_conditions, 0].unsqueeze(1).to(self.device)

                    # update generator every n iterations as suggested in paper
                    if i_batch % self.critic_iterations == 0:
                        train_generator = True
                    else:
                        train_generator = False

                    d_loss, g_loss, gen_samples_batch = self.batch_train(data, data_labels, train_generator)

                    d_loss_batch += d_loss
                    g_loss_batch += g_loss
                    i_batch += 1
                self.d_losses.append(d_loss_batch/i_batch)
                self.g_losses.append(g_loss_batch/i_batch)

                # Save a checkpoint of the trained GAN and the generated samples every sample interval
                if epoch % self.sample_interval == 0:
                    gen_samples.append(gen_samples_batch[np.random.randint(0, len(batch))].detach().cpu().numpy())
                    # save models and optimizer states as checkpoints
                    # toggle between checkpoint files to avoid corrupted file during training
                    if trigger_checkpoint_01:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), samples=gen_samples)
                        trigger_checkpoint_01 = False
                    else:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), samples=gen_samples)
                        trigger_checkpoint_01 = True

                self.trained_epochs += 1
                loop.set_postfix_str(f"D LOSS: {np.round(d_loss_batch/i_batch,6)}, G LOSS: {np.round(g_loss_batch/i_batch,6)}")
        except KeyboardInterrupt:
            # save model at KeyboardInterrupt
            print("Keyboard interrupt detected.\nCancel training and continue with further operations.")

        self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file], samples=gen_samples, update_history=True)

        if isinstance(self.discriminator, EncoderDiscriminator):
            self.discriminator.encode_input()

        if isinstance(self.generator, DecoderGenerator):
            self.generator.decode_output()

        return gen_samples

    def batch_train(self, data, data_labels, train_generator):
        """Trains the GAN-Model on one batch of data.
        No further batch-processing. Give batch as to-be-used."""
        gen_cond_data_orig = None

        batch_size = data.shape[0]

        # gen_cond_data for prediction purposes; implemented but not tested right now;
        gen_cond_data = data[:, :self.input_sequence_length, :].to(self.device)

        # Channel recovery roughly implemented
        if self.input_sequence_length == self.sequence_length and self.n_channels > 1:
            recovery = 0.3
            zero_index = np.random.randint(0, self.n_channels, np.max((1, int(self.n_channels*recovery))))
            gen_cond_data[:, :, zero_index] = 0

        # if self.generator is instance of EncoderGenerator encode gen_cond_data to speed up training
        if isinstance(self.generator, DecoderGenerator) and self.input_sequence_length != 0:
            gen_cond_data_orig = gen_cond_data
            gen_cond_data = torch.cat((torch.zeros((batch_size, self.sequence_length - self.input_sequence_length, self.n_channels)).to(self.device), gen_cond_data), dim=1)
            gen_cond_data = self.generator.decoder.decode(gen_cond_data)

        seq_length = max(1, gen_cond_data.shape[1])
        gen_labels = torch.cat((gen_cond_data, data_labels.repeat(1, seq_length, 1).to(self.device)), dim=-1).to(self.device) if self.input_sequence_length != 0 else data_labels
        disc_labels = data_labels

        with torch.no_grad():
            # A. Riduzione temporale (es. 128 -> 64) tramite Encoder AE
            if isinstance(self.discriminator, EncoderDiscriminator):
                real_data_reduced = self.discriminator.encoder.encode(data)
            else:
                real_data_reduced = data

            # B. Preparazione per l'encoder head (aggiunta classi ai canali: 8 -> 9)
            # Questo garantisce che z_real sia estratto conoscendo la condizione
            real_data_with_cond = self.make_fake_data(real_data_reduced, data_labels)

            # C. ENCODING: estraiamo lo z reale [B, latent_dim] (es. 128)
            z_real = self.latent_encoder(real_data_with_cond)
            # In batch_train, prima di z_real = self.latent_encoder(...)
            # D. PERTURBAZIONE: z = z_reale + ϵ · N(0, σ^2)
            epsilon = self.epsilon if hasattr(self, 'epsilon') else 0.05
            
            # Usiamo il TUO metodo ufficiale per generare il rumore N(0, 1)
            # Il metodo restituisce [B, 1, latent_dim], quindi facciamo .squeeze(1) per avere [B, latent_dim]
            noise = self.sample_latent_variable(
                batch_size=batch_size, 
                latent_dim=self.latent_dim, 
                sequence_length=1, 
                device=self.device
            ).squeeze(1)
            # Applicazione Eq. 7: Somma del rumore al vettore latente reale
            z = z_real + epsilon * noise

        # 2. PREPARAZIONE INPUT GENERATORE
        # Concateniamo lo z perturbato con le labels (z_final diventa dimensione 129)
        # gen_labels (già calcolato nel tuo trainer) viene appiattito per la concatenazione
        z_final = torch.cat((z, gen_labels.view(batch_size, -1)), dim=-1).to(self.device)
        
        # Abilitiamo i gradienti per z_final affinché il Generatore riceva il segnale di errore
        z_final.requires_grad = True
        # -----------------
        #  Train Generator
        # -----------------
        if self.trained_epochs == 0: # Stampa solo al primo batch
                print(f"\n[DEBUG CANALI]")
                print(f"Shape segnale ridotto (EEG): {real_data_reduced.shape}")
                
                # Se stai usando make_fake_data prima dell'encoder:
                real_with_cond = self.make_fake_data(real_data_reduced, data_labels)
                print(f"Shape segnale passato all'Encoder: {real_with_cond.shape}")
                z_std = z_real.std().item()
                z_max = z_real.abs().max().item()
                
                # Calcoliamo la "potenza" della perturbazione che stai aggiungendo
                noise_scaled = self.epsilon * noise
                n_std = noise_scaled.std().item()
                
                print(f"\n[DEBUG MAGNITUDO Z]")
                print(f"Ampiezza Z_REAL: std={z_std:.4f}, max={z_max:.4f}")
                print(f"Ampiezza PERTURBAZIONE (eps*N): std={n_std:.4f}")
                print(f"Rapporto Rumore/Segnale: {(n_std/z_std)*100:.2f}%")
                print(f"\n[DEBUG CLASS CONDITIONING]")
                print(f"Prime 5 label date al Generatore: {z_final[:5, -1]}")
                print(f"Prime 5 label REALI: {data_labels[:5, 0, 0]}")
        if train_generator:

            # enable training mode for generator; disable training mode for discriminator + freeze discriminator weights
            self.generator.train()
            self.discriminator.eval()

            # Sample noise and labels as generator input
            #z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim, sequence_length=seq_length, device=self.device)
            #z = torch.cat((z, gen_labels), dim=-1).to(self.device)
            #z.requires_grad = True
            
            # Generate a batch of samples
            gen_imgs = self.generator(z_final)

            if gen_cond_data_orig is not None:
                # gen_cond_data was encoded before; use the original form to make fake data
                fake_data = self.make_fake_data(gen_imgs, data_labels, gen_cond_data_orig)
            else:
                fake_data = self.make_fake_data(gen_imgs, data_labels, gen_cond_data)

            # Compute loss/validity of generated data and update generator
            validity = self.discriminator(fake_data)
            g_loss = self.loss.generator(validity)
            self.generator_optimizer.zero_grad()
            g_loss.backward()
            self.generator_optimizer.step()

            g_loss = g_loss.item()
            self.prev_g_loss = g_loss
        else:
            g_loss = self.prev_g_loss

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # enable training mode for discriminator; disable training mode for generator + freeze generator weights
        self.generator.eval()
        self.discriminator.train()

        # Create a batch of generated samples
        with torch.no_grad():
            # Sample noise and labels as generator input
            #z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim, sequence_length=seq_length, device=self.device)
            #z = torch.cat((z, gen_labels), dim=-1).to(self.device)

            # Generate a batch of fake samples
            gen_imgs = self.generator(z_final)
            if gen_cond_data_orig is not None:
                # gen_cond_data was encoded before; use the original form to make fake data
                fake_data = self.make_fake_data(gen_imgs, disc_labels, gen_cond_data_orig)
            else:
                fake_data = self.make_fake_data(gen_imgs, disc_labels, gen_cond_data)

            if self.trained_epochs % self.sample_interval == 0:
                # decode gen_imgs if necessary - decoding only necessary if not prediction case or seq2seq case
                if not hasattr(self.generator, 'module'):
                    decode_imgs = isinstance(self.generator, DecoderGenerator) and not self.generator.decode
                else:
                    decode_imgs = isinstance(self.generator.module, DecoderGenerator) and not self.generator.module.decode

                if decode_imgs:
                    if not hasattr(self.generator, 'module'):
                        fake_input = fake_data[:,:,:self.generator.channels].reshape(-1, self.generator.seq_len, self.generator.channels)
                        gen_samples = self.generator.decoder.decode(fake_input)
                    else:
                        fake_input = fake_data[:,:,:self.generator.module.channels].reshape(-1, self.generator.module.seq_len, self.generator.module.channels)
                        gen_samples = self.generator.module.decoder.decode(fake_input)
                    
                    # concatenate gen_cond_data_orig with decoded fake_data
                    # currently redundant because gen_cond_data is None in this case
                    if self.input_sequence_length != 0 and self.input_sequence_length != self.sequence_length:
                        if gen_cond_data_orig is not None:
                            gen_samples = torch.cat((gen_cond_data_orig, gen_samples), dim=1).to(self.device)
                        else:
                            gen_samples = torch.cat((gen_cond_data, gen_samples), dim=1).to(self.device)
                else:
                    gen_samples = fake_data[:, :, :self.n_channels]
                # concatenate channel names, conditions and generated samples
                gen_samples = torch.cat((data_labels.permute(0, 2, 1).repeat(1, 1, self.n_channels), gen_samples), dim=1)  # if self.n_conditions > 0 else gen_samples
            else:
                gen_samples = None

            if not hasattr(self.generator, 'module'):
                real_data = self.discriminator.encoder.encode(data) if isinstance(self.discriminator, EncoderDiscriminator) and not self.discriminator.encode else data
            else:
                real_data = self.discriminator.module.encoder.encode(data) if isinstance(self.discriminator.module, EncoderDiscriminator) and not self.discriminator.module.encode else data

            real_data = self.make_fake_data(real_data, disc_labels)

        # Loss for real and generated samples
        real_data.requires_grad = True
        fake_data.requires_grad = True
        validity_fake = self.discriminator(fake_data)
        validity_real = self.discriminator(real_data)

        # Total discriminator loss and update
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            d_loss = self.loss.discriminator(validity_real, validity_fake, self.discriminator, real_data, fake_data)
        else:
            d_loss = self.loss.discriminator(validity_real, validity_fake)
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        return d_loss.item(), g_loss, gen_samples

    def save_checkpoint(self, path_checkpoint=None, samples=None, generator=None, discriminator=None, update_history=False):
        if path_checkpoint is None:
            path_checkpoint = 'trained_models'+os.path.sep+'checkpoint.pt'
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        if update_history:
            self.configuration['trained_epochs'] = self.trained_epochs
            self.configuration['history']['trained_epochs'] = self.configuration['history']['trained_epochs'] + [self.trained_epochs]            
            self.configuration['train_time'] = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))

        state_dict = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'generator_loss': self.g_losses,
            'discriminator_loss': self.d_losses,
            'samples': samples,
            'trained_epochs': self.trained_epochs,
            'configuration': self.configuration,
        }
        torch.save(state_dict, path_checkpoint)

        if update_history:
            print(f"Checkpoint saved to {path_checkpoint}.")
            print(f"Training complete in: {self.configuration['train_time']}")

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            self.generator.load_state_dict(state_dict['generator'])
            self.discriminator.load_state_dict(state_dict['discriminator'])
            self.generator_optimizer.load_state_dict(state_dict['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])
            print(f"Device {self.device}:{self.rank}: Using pretrained GAN.")
        else:
            Warning("No checkpoint-file found. Using random initialization.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, generator=None, discriminator=None, samples=None, update_history=False):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), generator=generator, discriminator=discriminator, samples=samples, update_history=update_history)

        for f in checkpoint_files:
            if os.path.exists(os.path.join(path_checkpoint, f)):
                os.remove(os.path.join(path_checkpoint, f))

    def print_log(self, current_epoch, d_loss, g_loss):
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (current_epoch, self.epochs,
               d_loss, g_loss)
        )

    def set_optimizer_state(self, optimizer, g_or_d='G'):
        if g_or_d == 'G':
            self.generator_optimizer.load_state_dict(optimizer)
            print('Generator optimizer state loaded successfully.')
        elif g_or_d == 'D':
            self.discriminator_optimizer.load_state_dict(optimizer)
            print('Discriminator optimizer state loaded successfully.')
        else:
            raise ValueError('G_or_D must be either "G" (Generator) or "D" (Discriminator)')

    @staticmethod
    def sample_latent_variable(batch_size=1, latent_dim=1, sequence_length=1, device=torch.device('cpu')):
        """samples a latent variable from a normal distribution
        as a tensor of shape (batch_size, (sequence_length), latent_dim) on the given device"""
        return torch.randn((batch_size, sequence_length, latent_dim), device=device).float()

    def make_fake_data(self, gen_imgs, data_labels, condition_data=None):
        """
        :param gen_imgs: generated data from generator of shape (batch_size, sequence_length, n_channels)
        :param data_labels: scalar labels/conditions for generated data of shape (batch_size, 1, n_conditions)
        :param condition_data: additional data for conditioning the generator of shape (batch_size, input_sequence_length, n_channels)
        """
        # if input_sequence_length is available and self.generator is instance of DecoderGenerator
        # decode gen_imgs, concatenate with gen_cond_data_orig and encode again to speed up training
        if self.input_sequence_length != 0 and self.input_sequence_length != self.sequence_length:
            # prediction case
            if gen_imgs.shape[1] == self.sequence_length:
                gen_imgs = gen_imgs[:, self.input_sequence_length:, :]
            fake_data = torch.cat((condition_data, gen_imgs), dim=1).to(self.device)
        else:
            fake_data = gen_imgs
        if data_labels.shape[-1] != 0:
            # concatenate labels with fake data if labels are available
            fake_data = torch.cat((fake_data, data_labels.repeat(1, fake_data.shape[1], 1)), dim=-1).to(self.device)

        return fake_data


class AETrainer(Trainer):
    """Trainer for Autoencoder."""

    def __init__(self, model, opt):
        # training configuration
        super().__init__()
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank
        self.training_levels = opt['training_levels']
        self.training_level = opt['training_level']
        self.start_time = time.time()

        # model
        self.model = model
        self.model.to(self.device)

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()

        # training statistics
        self.trained_epochs = 0
        self.train_loss = []
        self.test_loss = []

        self.configuration = {
            'device': self.device,
            'model_class': str(self.model.__class__.__name__),
            'batch_size': self.batch_size,
            'n_epochs': self.epochs,
            'sample_interval': self.sample_interval,
            'learning_rate': self.learning_rate,
            'hidden_dim': opt['hidden_dim'],
            'data': opt['data'] if 'data' in opt else None,
            'checkpoint': opt['checkpoint'] if 'checkpoint' in opt else None,
            'channels_in': opt['channels_in'],
            'time_in': opt['time_in'],
            'time_out': opt['time_out'] if 'time_out' in opt else None,
            'channels_out': opt['channels_out'] if 'channels_out' in opt else None,
            'sequence_length': opt['sequence_length'],
            'target': opt['target'] if 'target' in opt else None,
            'trained_epochs': self.trained_epochs,
            'input_dim': opt['input_dim'],
            'output_dim': opt['output_dim'],
            'output_dim_2': opt['output_dim_2'],
            'num_layers': opt['num_layers'],
            'num_heads': opt['num_heads'],
            'seed': opt['seed'],
            'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
            'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
            'save_name': opt['save_name'] if 'save_name' in opt else '',
            'dataloader': {
                'data': opt['data'] if 'data' in opt else None,
                'diff_data': opt['diff_data'] if 'diff_data' in opt else None,
                'std_data': opt['std_data'] if 'std_data' in opt else None,
                'norm_data': opt['norm_data'] if 'norm_data' in opt else None,
                'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
                'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
                'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            },
            'history': opt['history'] if 'history' in opt else None,
        }

    def training(self, train_data, test_data):
        path_checkpoint = 'trained_ae'
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        trigger_checkpoint_01 = True
        checkpoint_01_file = 'checkpoint_01.pt'
        checkpoint_02_file = 'checkpoint_02.pt'

        samples = []

        loop = tqdm(range(self.epochs))
        
        # try/except for KeyboardInterrupt --> Abort training and save model
        try:
            for epoch in loop:
                train_loss, test_loss, sample = self.batch_train(train_data, test_data)
                self.train_loss.append(train_loss)
                self.test_loss.append(test_loss)

                loop.set_postfix_str(f"TRAIN LOSS: {np.round(train_loss,6)}, TEST LOSS: {np.round(test_loss,6)}")

                if len(sample) > 0:
                    samples.append(sample)

                # Save a checkpoint of the trained AE and the generated samples every sample interval
                if epoch % self.sample_interval == 0:
                    # save models and optimizer states as checkpoints
                    # toggle between checkpoint files to avoid corrupted file during training
                    if trigger_checkpoint_01:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), samples=samples)
                        trigger_checkpoint_01 = False
                    else:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), samples=samples)
                        trigger_checkpoint_01 = True

                self.trained_epochs += 1
        except KeyboardInterrupt:
            # save model at KeyboardInterrupt
            print("Keyboard interrupt detected.\nCancel training and continue with further operations.")

        self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file], update_history=True, samples=samples)
        return samples

    def batch_train(self, train_data, test_data):
        train_loss = self.train_model(train_data)
        test_loss, samples = self.test_model(test_data)
        return train_loss, test_loss, samples

    def train_model(self, data):
        self.model.train()
        total_loss = 0
        for batch in data:
            self.optimizer.zero_grad()
            inputs = batch.float().to(self.model.device)
            outputs = self.model(inputs)
            loss = self.loss(outputs, inputs)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data)

    def test_model(self, data):
        self.model.eval()
        total_loss = 0
        samples = []
        with torch.no_grad():
            for batch in data:
                inputs = batch.float().to(self.model.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, inputs)
                total_loss += loss.item()
                if self.trained_epochs % self.sample_interval == 0:
                    samples.append(np.stack([inputs.cpu().numpy(), outputs.cpu().numpy()], axis=1))
        if len(samples) > 0:
            samples = np.concatenate(samples, axis=0)[np.random.randint(0, len(samples))].reshape(1, *samples[0].shape[1:])
        return total_loss / len(data), samples

    def save_checkpoint(self, path_checkpoint=None, model=None, update_history=False, samples=None):
        if path_checkpoint is None:
            default_path = 'trained_ae'
            if not os.path.exists(default_path):
                os.makedirs(default_path)
            path_checkpoint = os.path.join(default_path, 'checkpoint.pt')

        if model is None:
            model = self.model

        if update_history:
            self.configuration['trained_epochs'] = self.trained_epochs
            self.configuration['history']['trained_epochs'] = self.configuration['history']['trained_epochs'] + [self.trained_epochs]
            self.configuration['train_time'] = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))

        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'trained_epochs': self.trained_epochs,
            'samples': samples,
            'configuration': self.configuration,
        }

        if self.training_levels == 2 and self.training_level == 2:
            checkpoint_dict['model_1'] = self.model1_states['model']
            checkpoint_dict['model_1_optimizer'] = self.model1_states['optimizer']

        if not (self.training_levels == 2 and self.training_level == 1) or 'checkpoint.pt' not in path_checkpoint:
            torch.save(checkpoint_dict, path_checkpoint)
            
            if update_history:
                print(f"Checkpoint saved to {path_checkpoint}.")
                print(f"Training complete in: {self.configuration['train_time']}")

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.') 
            if self.training_levels == 2 and self.training_level == 1:
                self.model.load_state_dict(state_dict['model_1'])
                self.optimizer.load_state_dict(state_dict['model_1_optimizer'])
            else:           
                self.model.load_state_dict(state_dict['model'])
                self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            raise FileNotFoundError(f"Checkpoint-file {path_checkpoint} was not found.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, model=None, update_history=False, samples=None):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), model=None, update_history=update_history, samples=samples)

        for f in checkpoint_files:
            if os.path.exists(os.path.join(path_checkpoint, f)):
                os.remove(os.path.join(path_checkpoint, f))

    def print_log(self, current_epoch, train_loss, test_loss):
        print(
            "[Epoch %d/%d] [Train loss: %f] [Test loss: %f]" % (current_epoch, self.epochs, train_loss, test_loss)
        )

    def set_optimizer_state(self, optimizer):
        self.optimizer.load_state_dict(optimizer)
        print('Optimizer state loaded successfully.')

class VAETrainer(Trainer):
    """Trainer for VAE"""

    def __init__(self, model, opt):
        # training configuration
        super().__init__()
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank
        self.kl_alpha = opt['kl_alpha'] if 'kl_alpha' in opt else .00001
        self.n_conditions = len(opt['kw_conditions']) if 'kw_conditions' in opt else 0
        self.start_time = time.time()

        # model
        self.model = model
        self.model.to(self.device)

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()

        # training statistics
        self.trained_epochs = 0
        self.train_loss = []

        self.configuration = {
            'device': self.device,
            'model_class': str(self.model.__class__.__name__),
            'batch_size': self.batch_size,
            'n_epochs': self.epochs,
            'sample_interval': self.sample_interval,
            'learning_rate': self.learning_rate,
            'hidden_dim': opt['hidden_dim'],
            'encoded_dim': opt['encoded_dim'],
            'path_dataset': opt['path_dataset'] if 'path_dataset' in opt else None,
            'path_checkpoint': opt['path_checkpoint'] if 'path_checkpoint' in opt else None,
            'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
            'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
            'trained_epochs': self.trained_epochs,
            'input_dim': opt['input_dim'],
            'save_name': opt['save_name'] if 'save_name' in opt else '',
            'activation': opt['activation'] if 'activation' in opt else 'tanh',
            'dataloader': {
                'data': opt['data'] if 'data' in opt else None,
                'diff_data': opt['diff_data'] if 'diff_data' in opt else None,
                'std_data': opt['std_data'] if 'std_data' in opt else None,
                'norm_data': opt['norm_data'] if 'norm_data' in opt else None,
                'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
                'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
                'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            },
            'history': opt['history'] if 'history' in opt else None,
        }

    def training(self, dataset: DataLoader):
        try:
            self.recon_losses = []
            self.kl_losses = []
            self.losses = []
            gen_samples = []
            
            path_checkpoint = 'trained_vae'
            if not os.path.exists(path_checkpoint):
                os.makedirs(path_checkpoint)
            trigger_checkpoint_01 = True
            checkpoint_01_file = 'checkpoint_01.pt'
            checkpoint_02_file = 'checkpoint_02.pt'
            
            loop = tqdm(range(self.epochs))
            for epoch in loop:
                self.epoch = epoch
                epoch_loss = self.batch_train(dataset)
                self.train_loss.append(epoch_loss)
                loop.set_postfix(loss=self.batch_loss.item())

                #Generate samples on interval
                if self.epoch % self.sample_interval == 0:
                    generated_samples = torch.Tensor(self.model.generate_samples(loader=dataset,condition=0,num_samples=1000)).to(self.device)
                    gen_samples.append(generated_samples[np.random.randint(0, generated_samples.shape[0])].detach().tolist()) #TODO: Not sure if this is the same as the GAN

                    # save models and optimizer states as checkpoints
                    # toggle between checkpoint files to avoid corrupted file during training
                    if trigger_checkpoint_01:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), samples=gen_samples)
                        trigger_checkpoint_01 = False
                    else:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), samples=gen_samples)
                        trigger_checkpoint_01 = True
                    
                self.trained_epochs += 1

            self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file], update_history=True, samples=gen_samples)
            
            return gen_samples

        except KeyboardInterrupt:
            # save model at KeyboardInterrupt
            print("Keyboard interrupt detected.\nSaving checkpoint...")
            self.save_checkpoint(update_history=True, samples=gen_samples)

    def batch_train(self, data):

        self.model.train()
        total_loss = 0
        for batch in data:

            #Run data through model
            inputs = batch[:,self.n_conditions:,:].to(self.model.device)
            x_reconstruction, mu, sigma = self.model(inputs)
            
            #Loss
            reconstruction_loss = self.loss(x_reconstruction, inputs)
            kl_div = torch.mean(-0.5 * torch.sum(1 + sigma - mu**2 - torch.exp(sigma), axis=1), dim=0)
            self.batch_loss = reconstruction_loss + kl_div*self.kl_alpha

            #Update
            self.optimizer.zero_grad()
            self.batch_loss.backward()
            self.optimizer.step()
            total_loss += self.batch_loss.item()

        self.recon_losses.append(reconstruction_loss.detach().tolist())
        self.kl_losses.append(kl_div.detach().tolist())
        self.losses.append(self.batch_loss.detach().tolist())

        return total_loss / len(data)

    def save_checkpoint(self, path_checkpoint=None, model=None, update_history=False, samples=None):
        if path_checkpoint is None:
            default_path = 'trained_ae'
            if not os.path.exists(default_path):
                os.makedirs(default_path)
            path_checkpoint = os.path.join(default_path, 'checkpoint.pt')

        if model is None:
            model = self.model

        if update_history:
            self.configuration['trained_epochs'] = self.trained_epochs
            self.configuration['history']['trained_epochs'] = self.configuration['history']['trained_epochs'] + [self.trained_epochs]
            self.configuration['train_time'] = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))
        
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'trained_epochs': self.trained_epochs,
            'samples': samples,
            'configuration': self.configuration,
        }

        torch.save(checkpoint_dict, path_checkpoint)

        if update_history:
            print(f"Checkpoint saved to {path_checkpoint}.")
            print(f"Training complete in: {self.configuration['train_time']}")

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.')
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            raise FileNotFoundError(f"Checkpoint-file {path_checkpoint} was not found.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, model=None, update_history=False, samples=None):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), model=None, update_history=update_history, samples=samples)

        for f in checkpoint_files:
            if os.path.exists(os.path.join(path_checkpoint, f)):
                os.remove(os.path.join(path_checkpoint, f))

    def print_log(self, current_epoch, train_loss, test_loss):
        print(
            "[Epoch %d/%d] [Train loss: %f] [Test loss: %f]" % (current_epoch, self.epochs, train_loss, test_loss)
        )

    def set_optimizer_state(self, optimizer):
        self.optimizer.load_state_dict(optimizer)
        print('Optimizer state loaded successfully.')


#LateneAETrainer inherits from Trainer and implements the training loop for a Latent Autoencoder, which is a specific type of autoencoder that learns a latent representation of the input data. The training loop includes both the reconstruction loss and the KL divergence loss, which encourages the latent space to follow a specific distribution (usually a normal distribution). The trainer also manages checkpoints and logs training progress.
class LatentAETrainer(Trainer):
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt):
        # training configuration
        super().__init__()
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = opt['sequence_length'] if 'sequence_length' in opt else 0
        self.input_sequence_length = opt['input_sequence_length'] if 'input_sequence_length' in opt else 0
        self.sequence_length_generated = self.sequence_length
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.latent_dim = opt['latent_dim'] if 'latent_dim' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.ae_lr = opt['ae_lr'] if 'learning_rate' in opt else 0.0001
        self.g_lr = self.ae_lr
        self.d_lr = self.ae_lr
        self.n_conditions = opt['n_conditions'] if 'n_conditions' in opt else 0
        self.n_channels = opt['n_channels'] if 'n_channels' in opt else 1
        self.channel_names = opt['channel_names'] if 'channel_names' in opt else list(range(0, self.n_channels))
        self.b1, self.b2 = 0, 0.9  # alternative values: .5, 0.999
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank
        self.start_time = time.time()

        self.generator = generator
        self.discriminator = discriminator
        if hasattr(generator, 'module'):
            self.generator = {k.partition('module.')[2]: v for k,v in generator}
            self.discriminator = {k.partition('module.')[2]: v for k,v in discriminator}

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr=self.g_lr, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                lr=self.d_lr, betas=(self.b1, self.b2))

        self.criterion = nn.SmoothL1Loss()
        #self.loss = Loss()
        #if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
         #   self.loss.set_lambda_gp(self.lambda_gp)

        self.d_losses = []
        self.g_losses = []
        self.trained_epochs = 0

        #self.prev_g_loss = 0
        generator_class = str(self.generator.__class__.__name__) if not isinstance(self.generator, DecoderGenerator) else str(self.generator.generator.__class__.__name__)
        discriminator_class = str(self.discriminator.__class__.__name__) if not isinstance(self.discriminator, EncoderDiscriminator) else str(self.discriminator.discriminator.__class__.__name__)
        # Gestione Discriminatore (Encoder)
        if isinstance(self.discriminator, EncoderDiscriminator):
            base_disc = self.discriminator.discriminator
        else:
            base_disc = self.discriminator

        # Definizione teste per il Latent AE
        self.encoder_head = nn.Sequential(
            base_disc[0],             # Patch Embedding
            base_disc[1],             # Transformer Blocks
            base_disc[2].clshead[0],  # Reduce (Mean Pooling)
            base_disc[2].clshead[1]   # LayerNorm
        )
        self.encoder_head.to(self.device)

        # FIX 3: Aggiornamento configurazione per LatentAE
        self.configuration = {
            'device': self.device,
            'generator_class': generator_class,
            'discriminator_class': discriminator_class,
            'model_class': 'LatentAE', # Identificativo corretto
            'sequence_length': self.sequence_length,
            'sequence_length_generated': self.sequence_length_generated,
            'num_layers': opt['num_layers'],
            'hidden_dim': opt['hidden_dim'],
            'latent_dim': opt['hidden_dim'],
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'trained_epochs': self.trained_epochs,
            'sample_interval': self.sample_interval,
            'ae_lr': self.ae_lr,
            'n_conditions': self.n_conditions,
            'patch_size': opt['patch_size'] if 'patch_size' in opt else None,
            'b1': self.b1,
            'b2': self.b2,
            'data': opt['data'] if 'data' in opt else None,
            'autoencoder': opt['autoencoder'] if 'autoencoder' in opt else None,
            'n_channels': self.n_channels,
            'channel_names': self.channel_names,
            'seed': opt['seed'],
            'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
            'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
            'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            'save_name': opt['save_name'] if 'save_name' in opt else '',
            'dataloader': {
                'data': opt['data'] if 'data' in opt else None,
                'kw_conditions': opt['kw_conditions'] if 'kw_conditions' in opt else None,
                'diff_data': opt['diff_data'] if 'diff_data' in opt else None,
                'std_data': opt['std_data'] if 'std_data' in opt else None,
                'norm_data': opt['norm_data'] if 'norm_data' in opt else None,
                'kw_time': opt['kw_time'] if 'kw_time' in opt else None,
                'kw_channel': opt['kw_channel'] if 'kw_channel' in opt else None,
            },
            'history': opt['history'] if 'history' in opt else {},
        }

    def training(self, dataset: DataLoader):
            """Batch training of the Latent Autoencoder (L2 Reconstruction)."""
            gen_samples = []
            # Percorsi per i checkpoint (fedele all'originale)
            path_checkpoint = 'trained_models'
            trigger_checkpoint_01 = True
            checkpoint_01_file = 'checkpoint_01.pt'
            checkpoint_02_file = 'checkpoint_02.pt'

            gen_samples_batch = None
            batch = None

            loop = tqdm(range(self.epochs))

            try:
                for epoch in loop:
                    i_batch = 0
                    ae_loss_batch = 0 # Loss di ricostruzione cumulata

                    for batch in dataset:
                        # Estrazione dati conforme alla repository: [B, Time, Chan]
                        # I primi n_conditions punti temporali contengono le labels float
                        data = batch[:, self.n_conditions:].to(self.device)
                        data_labels = batch[:, :self.n_conditions, 0].unsqueeze(1).to(self.device)

                        # Nell'Autoencoder aggiorniamo i pesi ad ogni batch (collaborazione)
                        # Non serve la logica 'critic_iterations' della GAN
                        # Restituiamo la loss L2 e i campioni ricostruiti
                        loss, _, gen_samples_batch = self.batch_train(data, data_labels)

                        ae_loss_batch += loss
                        i_batch += 1

                    # Registriamo la loss media dell'epoca (fedele alla struttura d_losses/g_losses)
                    current_epoch_loss = ae_loss_batch / i_batch
                    self.d_losses.append(current_epoch_loss)
                    self.g_losses.append(current_epoch_loss)

                    # Salvataggio checkpoint ogni 'sample_interval' (fedele all'originale)
                    if epoch % self.sample_interval == 0:
                        # Scegliamo un campione casuale per visualizzare la qualità della ricostruzione
                        sample_idx = np.random.randint(0, len(batch))
                        gen_samples.append(gen_samples_batch[sample_idx].detach().cpu().numpy())

                        if trigger_checkpoint_01:
                            self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), samples=gen_samples)
                            trigger_checkpoint_01 = False
                        else:
                            self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), samples=gen_samples)
                            trigger_checkpoint_01 = True

                    self.trained_epochs += 1
                    # Aggiorniamo il progresso con la loss L2 (indica quanto bene stiamo ricostruendo)
                    loop.set_postfix_str(f"RECON LOSS (L2): {np.round(current_epoch_loss, 6)}")

            except KeyboardInterrupt:
                print("Keyboard interrupt detected.\nSaving current state and continuing.")

            # Gestione finale dei checkpoint (fedele all'originale)
            self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file], samples=gen_samples, update_history=True)

            # Attivazione dei wrapper per l'utilizzo post-training (fondamentale!)
            if isinstance(self.discriminator, EncoderDiscriminator):
                self.discriminator.encode_input()

            if isinstance(self.generator, DecoderGenerator):
                self.generator.decode_output()

            return gen_samples

    def batch_train(self, data, data_labels):
            """
            Trains the Latent Autoencoder on one batch of data.
            Faithful to the repository structure and wrapper logic.
            """
            batch_size = data.shape[0]

            # Mettiamo entrambi i modelli in training mode (collaborazione)
            self.generator.train()
            self.discriminator.train()

            # 1. COMPRESSIONE TEMPORALE (Fedele al Discriminator Wrapper)
            # Se il discriminatore è un EncoderDiscriminator, passiamo i dati reali
            # attraverso l'autoencoder temporale per ottenere la rappresentazione ridotta.
            if isinstance(self.discriminator, EncoderDiscriminator) and self.discriminator.encode:
                real_data_reduced = self.discriminator.encoder.encode(data)
            else:
                real_data_reduced = data

            # 2. ENCODING LATENTE (Encoder part)
            # Usiamo la encoder_head definita nell'__init__ per mappare
            # il segnale ridotto in un vettore latente z.
            # real_data_reduced: [B, Chan, Seq_Small] -> z: [B, latent_dim]
            real_data_with_cond = self.make_fake_data(real_data_reduced, data_labels)
            z = self.encoder_head(real_data_with_cond)
            cond = data_labels.view(batch_size, -1)
            z_conditioned = torch.cat((z, cond), dim=-1)
            # 3. RICOSTRUZIONE (Generator part)
            # Usiamo il modello base del generatore (TTS) per ricostruire il segnale ridotto.
            if isinstance(self.generator, DecoderGenerator):
                recon_data_reduced = self.generator.generator(z_conditioned)
            else:
                recon_data_reduced = self.generator(z_conditioned)

            # Allineamento shape: Il TTSGenerator sputa [B, Seq, Chan],
            # l'AE temporale lavora con [B, Chan, Seq].
            if recon_data_reduced.shape[1] != real_data_reduced.shape[1]:
                recon_data_reduced = recon_data_reduced.permute(0, 2, 1).contiguous()

            # 4. CALCOLO LOSS L2 (MSE)
            # Calcoliamo la distanza tra il segnale compresso originale e quello ricostruito.
            loss = self.criterion(recon_data_reduced, real_data_reduced)

            # 5. AGGIORNAMENTO PESI
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()
            self.discriminator_optimizer.step()

            # 6. GENERAZIONE CAMPIONI PER VISUALIZZAZIONE (Fedele alla logica GAN)
            # Se richiesto dal campionamento, riportiamo il segnale ricostruito
            # nello spazio temporale originale usando il decoder.
            with torch.no_grad():
                if self.trained_epochs % self.sample_interval == 0:
                    if isinstance(self.generator, DecoderGenerator) and self.generator.decode:
                        # Decoding: [B, Chan, Seq_Small] -> [B, Time_Orig, Chan]
                        gen_samples = self.generator.decoder.decode(recon_data_reduced)
                    else:
                        gen_samples = recon_data_reduced

                    # Concateniamo le label float all'inizio (come richiesto dalla repository)
                    # per mantenere la compatibilità con i campioni reali.
                    gen_samples = torch.cat((
                        data_labels.permute(0, 2, 1).repeat(1, 1, self.n_channels),
                        gen_samples
                    ), dim=1)
                else:
                    gen_samples = None

            # Restituiamo la loss (duplicata per g_loss e d_loss per compatibilità history)
            return loss.item(), loss.item(), gen_samples
    def save_checkpoint(self, path_checkpoint=None, samples=None, generator=None, discriminator=None, update_history=False):
        if path_checkpoint is None:
            path_checkpoint = 'trained_models'+os.path.sep+'checkpoint.pt'
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        if update_history:
            self.configuration['trained_epochs'] = self.trained_epochs
            self.configuration['history']['trained_epochs'] = self.configuration['history']['trained_epochs'] + [self.trained_epochs]            
            self.configuration['train_time'] = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))

        state_dict = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'generator_loss': self.g_losses,
            'discriminator_loss': self.d_losses,
            'samples': samples,
            'trained_epochs': self.trained_epochs,
            'configuration': self.configuration,
        }
        torch.save(state_dict, path_checkpoint)

        if update_history:
            print(f"Checkpoint saved to {path_checkpoint}.")
            print(f"Training complete in: {self.configuration['train_time']}")

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            self.generator.load_state_dict(state_dict['generator'])
            self.discriminator.load_state_dict(state_dict['discriminator'])
            self.generator_optimizer.load_state_dict(state_dict['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])
            print(f"Device {self.device}:{self.rank}: Using pretrained GAN.")
        else:
            Warning("No checkpoint-file found. Using random initialization.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, generator=None, discriminator=None, samples=None, update_history=False):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), generator=generator, discriminator=discriminator, samples=samples, update_history=update_history)

        for f in checkpoint_files:
            if os.path.exists(os.path.join(path_checkpoint, f)):
                os.remove(os.path.join(path_checkpoint, f))

    def print_log(self, current_epoch, d_loss, g_loss):
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (current_epoch, self.epochs,
               d_loss, g_loss)
        )

    def set_optimizer_state(self, optimizer, g_or_d='G'):
        if g_or_d == 'G':
            self.generator_optimizer.load_state_dict(optimizer)
            print('Generator optimizer state loaded successfully.')
        elif g_or_d == 'D':
            self.discriminator_optimizer.load_state_dict(optimizer)
            print('Discriminator optimizer state loaded successfully.')
        else:
            raise ValueError('G_or_D must be either "G" (Generator) or "D" (Discriminator)')

    @staticmethod
    def sample_latent_variable(batch_size=1, latent_dim=1, sequence_length=1, device=torch.device('cpu')):
        """samples a latent variable from a normal distribution
        as a tensor of shape (batch_size, (sequence_length), latent_dim) on the given device"""
        return torch.randn((batch_size, sequence_length, latent_dim), device=device).float()

    def make_fake_data(self, gen_imgs, data_labels, condition_data=None):
        """
        :param gen_imgs: generated data from generator of shape (batch_size, sequence_length, n_channels)
        :param data_labels: scalar labels/conditions for generated data of shape (batch_size, 1, n_conditions)
        :param condition_data: additional data for conditioning the generator of shape (batch_size, input_sequence_length, n_channels)
        """
        # if input_sequence_length is available and self.generator is instance of DecoderGenerator
        # decode gen_imgs, concatenate with gen_cond_data_orig and encode again to speed up training
        if self.input_sequence_length != 0 and self.input_sequence_length != self.sequence_length:
            # prediction case
            if gen_imgs.shape[1] == self.sequence_length:
                gen_imgs = gen_imgs[:, self.input_sequence_length:, :]
            fake_data = torch.cat((condition_data, gen_imgs), dim=1).to(self.device)
        else:
            fake_data = gen_imgs
        if data_labels.shape[-1] != 0:
            # concatenate labels with fake data if labels are available
            fake_data = torch.cat((fake_data, data_labels.repeat(1, fake_data.shape[1], 1)), dim=-1).to(self.device)

        return fake_data