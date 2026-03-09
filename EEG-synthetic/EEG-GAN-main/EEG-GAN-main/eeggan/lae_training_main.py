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
from eeggan.helpers.trainer import GANTrainer, LatentAETrainer
from eeggan.helpers.get_master import find_free_port
from eeggan.helpers.ddp_training import run, GANDDPTrainer
from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.initialize_gan import init_gan
from eeggan.helpers import system_inputs
from eeggan.helpers.system_inputs import default_inputs_lae_training
from eeggan.nn_architecture.models import EncoderDiscriminator, DecoderGenerator

def main(args=None):
    """Main function of the training process.
    For input help use the command 'python gan_training_main.py help' in the terminal."""
    print(">>> MAIN AVVIATO <<<")
    current_kw_dict = default_inputs_lae_training()
    
    if args is None:
        default_args = system_inputs.parse_arguments(sys.argv, kw_dict=current_kw_dict)
    else:
        default_args = system_inputs.parse_arguments(args, kw_dict=current_kw_dict)

    # --- LOGICA PERCORSO AUTOMATICO ---
    filename = default_args['data'].strip("'").strip('"')
    # Se l'utente scrive solo il nome del file, cercalo nella cartella del progetto
    if not filename.startswith('/'):
        default_args['data'] = os.path.join('/home/giorgio99/gan_bci', filename)
    
    print(f">>> CERCO IL FILE IN: {default_args['data']}")
    if not os.path.exists(default_args['data']):
        raise FileNotFoundError(f"Impossibile trovare il file: {default_args['data']}")
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
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
        'ae_lr': default_args['ae_lr'],
        'sample_interval': default_args['sample_interval'],
        'n_conditions': len(default_args['kw_conditions']) if default_args['kw_conditions'][0] != '' else 0,
        'patch_size': default_args['patch_size'],
        'kw_time': default_args['kw_time'],
        'kw_conditions': default_args['kw_conditions'],
        'sequence_length': -1,
        'input_sequence_length': 0,
        'num_layers': default_args['num_layers'],
        'latent_dim': default_args['hidden_dim'],
        'hidden_dim': default_args['hidden_dim'],
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
    print("\n--- DEBUG DIMENSIONI DATASET ---")
    print(f"Shape totale del dataset caricato: {dataset.shape}") # Dovrebbe essere [N, 128, Canali]
    print(f"Shape delle etichette (labels): {opt['n_conditions']}") # Quanti punti identifica come label?
    
    opt['channel_names'] = dataloader.channels
    opt['n_channels'] = dataset.shape[-1]
    opt['sequence_length'] = dataset.shape[1] - opt['n_conditions'] # escludiamo i punti delle label dalla lunghezza della sequenza
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
    generator, discriminator = init_gan(**opt)

    if isinstance(discriminator, EncoderDiscriminator):
        discriminator.encode_input(True)  # Forza la compressione da 128 a 64
        print("Wrapper Discriminatore FORZATO a True")
        
    if isinstance(generator, DecoderGenerator):
        generator.decode_output(True)   # Forza il ritorno a 128 per la visualizzazione
        print("Wrapper Generatore FORZATO a True")
    print("Generator and discriminator initialized.")

    # --------------------------------------------------------------------------------
    # Setup History
    # --------------------------------------------------------------------------------

    # Populate model configuration
    history = {}
    for key in opt.keys():
        if (not key == 'history') | (not key == 'trained_epochs'):
            history[key] = [opt[key]]
    history['trained_epochs'] = []

    if default_args['checkpoint'] != '':

        # load checkpoint
        model_dict = torch.load(default_args['checkpoint'])

        # update history
        for key in history.keys():
            history[key] = model_dict['configuration']['history'][key] + history[key]

    opt['history'] = history

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    # GAN-Training
    print('\n-----------------------------------------')
    print("Training Latent AE...")
    print('-----------------------------------------\n')
    if ddp:
            # Se implementi la versione DDP per l'AE
            # trainer = LatentAEDDPTrainer(generator, discriminator, opt)
            # mp.spawn(run_ae, ...)
            print("DDP for Latent AE not implemented in this snippet. falling back to single GPU.")
            ddp = False

    if not ddp:
            # 1. Inizializzazione Trainer specifico per AE (L2 Loss)
            trainer = LatentAETrainer(generator, discriminator, opt)

            if default_args['checkpoint'] != '':
                trainer.load_checkpoint(default_args['checkpoint'])

            # 2. GESTIONE BILANCIAMENTO CLASSI (WeightedRandomSampler)
            # Estraiamo le label dai dati reali (primo punto temporale, primo canale)
            # dataset shape: [N, Time, Channels]
            print("Calculating weights for class balancing...")
            label_indices = dataset[:, 0, 0].long() # Converte i float della classe in indici
            class_sample_count = torch.bincount(label_indices)
            weights = 1. / class_sample_count.float()
            samples_weights = torch.tensor([weights[t] for t in label_indices])

            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

            # 3. Creazione DataLoader Bilanciato (Invece di shuffle=True)
            dataset = DataLoader(dataset, batch_size=trainer.batch_size, sampler=sampler, pin_memory=True)

            # 4. Esecuzione Training (L2 Loss)
            ae_history = trainer.training(dataset)

            # --------------------------------------------------------------------------------
            # Save Results (Fedele allo standard del repository)
            # --------------------------------------------------------------------------------
            path = 'trained_models'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if opt['save_name'] != '':
                if not opt['save_name'].endswith('.pt'):
                    opt['save_name'] += '.pt'
                filename = opt['save_name']
            else:
                filename = f'latent_ae_{trainer.epochs}ep_' + timestamp + '.pt'

            path_checkpoint = os.path.join(path, filename)

            # Salviamo in un formato che la GAN possa leggere come checkpoint
            trainer.save_checkpoint(path_checkpoint=path_checkpoint, update_history=True)

            generator = trainer.generator
            discriminator = trainer.discriminator

            print(f"Latent AE training finished. Pre-trained weights saved at {path_checkpoint}")

            return generator, discriminator, opt, ae_history
if __name__ == '__main__':
    main()
