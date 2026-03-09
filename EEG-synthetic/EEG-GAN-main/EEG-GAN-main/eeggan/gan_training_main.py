import os
import sys
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler

# add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from eeggan.helpers.trainer import GANTrainer
from eeggan.helpers.get_master import find_free_port
from eeggan.helpers.ddp_training import run, GANDDPTrainer
from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.initialize_gan import init_gan
from eeggan.helpers import system_inputs

"""Implementation of the training process of a GAN for the generation of synthetic sequential data.

Instructions to start the training:
  - set the filename of the dataset to load
      - the shape of the dataset should be (n_samples, n_conditions + n_features)
      - the dataset should be a csv file
      - the first columns contain the conditions 
      - the remaining columns contain the time-series data
  - set the configuration parameters (Training configuration; Data configuration; GAN configuration)"""


def main(args=None):
    """Main function of the training process. 
    For input help use the command 'python gan_training_main.py help' in the terminal."""
    
    # create directory 'trained_models' if not exists
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
        print('Directory "../trained_models" created to store checkpoints and final model.')
    if args is None:
        default_args = system_inputs.parse_arguments(sys.argv, file='gan_training_main.py')
    else:
        default_args = system_inputs.parse_arguments(args, file='gan_training_main.py')

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    ddp = default_args['ddp']
    ddp_backend = "nccl" #default_args['ddp_backend']
    checkpoint = default_args['checkpoint']



    # Data configuration
    diff_data = False  # Differentiate data
    std_data = True  # Standardize data
    norm_data = False  # Normalize data

    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    if default_args['checkpoint'] != '':
        # check if checkpoint exists and otherwise take trained_models/checkpoint.pt
        if not os.path.exists(default_args['checkpoint']):
            print(f"Checkpoint {default_args['checkpoint']} does not exist. Checkpoint is set to 'trained_models/checkpoint.pt'.")
            default_args['checkpoint'] = os.path.join('trained_models', 'checkpoint.pt')
            checkpoint = default_args['checkpoint']
        print(f'Resuming training from checkpoint {checkpoint}.')

    # GAN configuration
    opt = {
        'n_epochs': default_args['n_epochs'],
        'checkpoint': default_args['checkpoint'],
        'data': default_args['data'],
        'autoencoder': default_args['autoencoder'],
        'batch_size': default_args['batch_size'],
        'discriminator_lr': default_args['discriminator_lr'],
        'generator_lr': default_args['generator_lr'],
        'sample_interval': default_args['sample_interval'],
        'n_conditions': len(default_args['kw_conditions']) if default_args['kw_conditions'][0] != '' else 0,
        'patch_size': default_args['patch_size'],
        'kw_time': default_args['kw_time'],
        'kw_conditions': default_args['kw_conditions'],
        'sequence_length': -1,
        'hidden_dim': default_args['hidden_dim'],  # Dimension of hidden layers in discriminator and generator
        'num_layers': default_args['num_layers'],
        'latent_dim': 128,  # Dimension of the latent space
        'critic_iterations': 5,  # number of iterations of the critic per generator iteration for Wasserstein GAN
        'lambda_gp': 10,  # Gradient penalty lambda for Wasserstein GAN-GP
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu") if not ddp else torch.device("cpu"), 
        'world_size': torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(),  # number of processes for distributed training
        'kw_channel': default_args['kw_channel'],
        'norm_data': norm_data,
        'std_data': std_data,
        'diff_data': diff_data,
        'seed': default_args['seed'],
        'save_name': default_args['save_name'],
        'history': None,
    }
    
    # set a seed for reproducibility if desired
    if opt['seed'] is not None:
        np.random.seed(opt['seed'])                       
        torch.manual_seed(opt['seed'])                    
        torch.cuda.manual_seed(opt['seed'])               
        torch.cuda.manual_seed_all(opt['seed'])           
        torch.backends.cudnn.deterministic = True  
    
    # Load dataset as tensor
    dataloader = Dataloader(default_args['data'],
                            kw_time=default_args['kw_time'],
                            kw_conditions=default_args['kw_conditions'],
                            norm_data=norm_data,
                            std_data=std_data,
                            diff_data=diff_data,
                            kw_channel=default_args['kw_channel'])
    dataset = dataloader.get_data()

    opt['channel_names'] = dataloader.channels
    opt['n_channels'] = dataset.shape[-1]
    opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]
    opt['n_samples'] = dataset.shape[0]

    ae_dict = torch.load(opt['autoencoder'], map_location=torch.device('cpu')) if opt['autoencoder'] != '' else []
    # check if generated sequence is a multiple of patch size   
    encoded_sequence = False
    def pad_warning(sequence_length, encoded_sequence=False):
        error_msg = f"Sequence length ({sequence_length}) must be a multiple of patch size ({default_args['patch_size']})."
        error_msg += " Please adjust the 'patch_size' or "
        if encoded_sequence:
            error_msg += "adjust the output sequence length of the autoencoder ('time_out'). The latter option requires a newly trained autoencoder."
        else:
            error_msg += "adjust the sequence length of the dataset."
        raise ValueError(error_msg)
    if ae_dict and (ae_dict['configuration']['target'] == 'full' or ae_dict['configuration']['target'] == 'time'):
        generated_seq_length = ae_dict['configuration']['time_out']
        encoded_sequence = True
    else:
        generated_seq_length = opt['sequence_length']
    if generated_seq_length % default_args['patch_size'] != 0:
        pad_warning(generated_seq_length, encoded_sequence)
        
    opt['latent_dim_in'] = opt['latent_dim'] + opt['n_conditions']
    opt['channel_in_disc'] = opt['n_channels'] + opt['n_conditions']
    opt['sequence_length_generated'] = opt['sequence_length']
    
    # --------------------------------------------------------------------------------
    # Initialize generator, discriminator and trainer
    # --------------------------------------------------------------------------------
    
    generator, discriminator = init_gan(**opt)

    # 2. Inizializzazione history locale basata sui parametri attuali
    history = {}
    for key in opt.keys():
        if (key != 'history') and (key != 'trained_epochs'):
            history[key] = [opt[key]]
    history['trained_epochs'] = [] 

    # 3. Caricamento Checkpoint e unione History ROBUSTA
    if default_args['checkpoint'] != '':
        print(f"Resuming training from checkpoint {default_args['checkpoint']}.")
        model_dict = torch.load(default_args['checkpoint'], map_location=opt['device'])

        # Carichiamo i pesi (Gestione chiavi LAE vs GAN)
        if 'generator' in model_dict: 
            generator.load_state_dict(model_dict['generator'])
            discriminator.load_state_dict(model_dict['discriminator'])
            print(">>> Pesi pre-train Latent AE caricati con successo!")
        elif 'generator_state_dict' in model_dict:
            generator.load_state_dict(model_dict['generator_state_dict'])
            discriminator.load_state_dict(model_dict['discriminator_state_dict'])
            print(">>> Pesi GAN caricati con successo!")

        # Unione history robusta: evita KeyError se mancano chiavi (es. discriminator_lr)
        if 'configuration' in model_dict and 'history' in model_dict['configuration']:
            old_history = model_dict['configuration']['history']
            for key in history.keys():
                if key in old_history:
                    history[key] = old_history[key] + history[key]
                else:
                    print(f">>> Info: Chiave '{key}' non presente nel checkpoint, inizializzata ex-novo.")

    # 4. Assegnazione history finale a opt
    opt['history'] = history
    print("Generator and discriminator initialized and history merged.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    # GAN-Training
    print('\n-----------------------------------------')
    print("Training GAN...")
    print('-----------------------------------------\n')
    if ddp:
        trainer = GANDDPTrainer(generator, discriminator, opt)
        if default_args['checkpoint'] != '':
            trainer.load_checkpoint(default_args['checkpoint'])
        mp.spawn(run,
                 args=(opt['world_size'], find_free_port(), ddp_backend, trainer, opt),
                 nprocs=opt['world_size'], join=True)
        
        print("GAN training finished.")
        
    else:
        trainer = GANTrainer(generator, discriminator, opt)
        if default_args['checkpoint'] != '':
            trainer.load_checkpoint(default_args['checkpoint'])

        n_cond = opt['n_conditions']

        if n_cond > 0:
            # 1. Recuperiamo le etichette (che sono float, es: 0.0, 1.0)
            # Prendiamo la prima etichetta prima del primo time point
            raw_labels = dataset[:, 0, 0].cpu().numpy() 

            # 2. TRUCCO FONDAMENTALE: np.unique con return_inverse
            # Questo trasforma [0.0, 1.0, 0.0] in [0, 1, 0] 
            # Ma gestisce bene anche se avessi [12.0, 15.0, 12.0] -> [0, 1, 0]
            unique_classes, labels_encoded = np.unique(raw_labels, return_inverse=True)
            
            # 3. Ora class_sample_count riceve interi (0, 1, 2...)
            class_sample_count = np.bincount(labels_encoded)
            
            # 4. Calcolo pesi (1 / frequenza)
            weights = 1. / class_sample_count

            samples_weight = weights[labels_encoded]
            samples_weight = torch.from_numpy(samples_weight).double()
                        # 5. Sampler e DataLoader
            sampler = WeightedRandomSampler(
                weights=samples_weight, 
                num_samples=len(samples_weight), 
                replacement=True
            )
            
            balanced_loader = DataLoader(
                dataset, 
                batch_size=trainer.batch_size, 
                sampler=sampler, 
                pin_memory=True
            )
            
            # Debug per essere sicuri
            print(f"\n[Sampler] Classi float rilevate: {unique_classes}")
            print(f"[Sampler] Conteggio per classe: {class_sample_count}")
            print(f"[Sampler] Pesi calcolati: {weights}\n")
            
        else:
            balanced_loader = DataLoader(
                dataset, 
                batch_size=trainer.batch_size, 
                shuffle=True, 
                pin_memory=True
            )
        #dataset = DataLoader(dataset, batch_size=trainer.batch_size, shuffle=True, pin_memory=True)
        gen_samples = trainer.training(balanced_loader)

        # save final models, optimizer states, generated samples, losses and configuration as final result
        path = 'trained_models'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if opt['save_name'] != '':
            # check if .pt extension is already included in the save_name
            if not opt['save_name'].endswith('.pt'):
                opt['save_name'] += '.pt'
            filename = opt['save_name']
        else:
            filename = f'gan_{trainer.epochs}ep_' + timestamp + '.pt'
        path_checkpoint = os.path.join(path, filename)
        trainer.save_checkpoint(path_checkpoint=path_checkpoint, samples=gen_samples, update_history=True)
        
        generator = trainer.generator
        discriminator = trainer.discriminator

        print("GAN training finished.")
        
        return generator, discriminator, opt, gen_samples


if __name__ == '__main__':
    main()
