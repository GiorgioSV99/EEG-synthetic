import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader

# add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from eeggan.helpers import system_inputs
from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.initialize_gan import init_gan
from eeggan.helpers.trainer import GANTrainer
from eeggan.nn_architecture.models import DecoderGenerator, EncoderDiscriminator
from eeggan.nn_architecture.vae_networks import VariationalAutoencoder

#another comment
def main(args=None):
    # 1. Ottieni il dizionario di default direttamente
    from eeggan.helpers.system_inputs import default_inputs_generate_lae
    current_kw_dict = default_inputs_generate_lae()

    # 2. Passa il dizionario DIRETTAMENTE al parser (usa kw_dict e togli file=...)
    if args is None:
        # Quando runni da terminale
        default_args = system_inputs.parse_arguments(sys.argv, kw_dict=current_kw_dict)
    else:
        # Quando viene chiamato dal centralino !eeggan
        default_args = system_inputs.parse_arguments(args, kw_dict=current_kw_dict)
    
    # 3. Ora usa default_args come al solito
    epsilon = default_args['epsilon']
    real_data_path = default_args['data']
    # set a seed for reproducibility if desired
    if default_args['seed'] is not None:
        np.random.seed(default_args['seed'])                       
        torch.manual_seed(default_args['seed'])                    
        torch.cuda.manual_seed(default_args['seed'])               
        torch.cuda.manual_seed_all(default_args['seed'])           
        torch.backends.cudnn.deterministic = True  
    
    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    num_samples_total = default_args['num_samples_total']
    num_samples_parallel = default_args['num_samples_parallel']

    condition = default_args['conditions']
    if not isinstance(condition, list):
        condition = [condition]
    # if no condition is given, make empty list
    if len(condition) == 1 and condition[0] == 'None':
        condition = []

    file = default_args['model']
    if file.split(os.path.sep)[0] == file and file.split('/')[0] == file:
        # use default path if no path is given
        path = 'trained_models'
        file = os.path.join(path, file)

    path_samples = default_args['save_name']
    if path_samples == '':
        # Use checkpoint filename as path
        path_samples = os.path.basename(file).split('.')[0] + '.csv'
    if path_samples.split(os.path.sep)[0] == path_samples:
        # use default path if no path is given
        path = 'generated_samples'
        if not os.path.exists(path):
            os.makedirs(path)
        path_samples = os.path.join(path, path_samples)

    state_dict = torch.load(file, map_location='cpu')

    # define device
    device = torch.device('cpu')

    # check if column condition labels are given
    n_conditions = len(state_dict['configuration']['kw_conditions']) if state_dict['configuration']['kw_conditions'] and state_dict['configuration']['kw_conditions'] != [''] else 0
    if n_conditions > 0:        
        col_labels = state_dict['configuration']['dataloader']['kw_conditions']
    else:
        col_labels = []
            
    # check if channel label is given
    if not state_dict['configuration']['dataloader']['kw_channel'] in [None, '']:
        kw_channel = [state_dict['configuration']['dataloader']['kw_channel']]
    else:
        kw_channel = ['Electrode']

    # get keyword for time step labels
    if state_dict['configuration']['dataloader']['kw_time']:
        kw_time = state_dict['configuration']['dataloader']['kw_time']
    else:
        kw_time = 'Time'

    real_dataloader = Dataloader(
        path=state_dict['configuration']['data'],
        kw_channel=state_dict['configuration']['dataloader']['kw_channel'] or 'Electrode',
        kw_conditions=state_dict['configuration']['dataloader']['kw_conditions'],
        kw_time=state_dict['configuration']['dataloader']['kw_time'],
        std_data=True # Fondamentale che sia lo stesso del training
    )
    real_dataset = real_dataloader.get_data() # Tensor [N, Time_Total, Chan]

    # Filtriamo i trial che appartengono alla condizione richiesta
    target_val = float(condition[0]) # Assumendo una sola condizione
    labels_all = real_dataset[:, 0, 0].numpy()
    target_indices = np.where(np.round(labels_all) == target_val)[0]
    filtered_real_data = real_dataset[target_indices]

    if state_dict['configuration']['model_class'] != 'VariationalAutoencoder':

        # load model/training configuration
        n_conditions = state_dict['configuration']['n_conditions']
        n_channels = state_dict['configuration']['n_channels']
        channel_names = state_dict['configuration']['channel_names']
        latent_dim = state_dict['configuration']['latent_dim']
        sequence_length = state_dict['configuration']['sequence_length']
        # input_sequence_length = state_dict['configuration']['input_sequence_length']

        if n_conditions != len(condition):
            raise ValueError(f"Number of conditions in model ({n_conditions}) does not match number of conditions given ({len(condition)}).")

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize generator
        print("Initializing generator...")
        latent_dim_in = latent_dim + n_conditions

        generator, discriminator = init_gan(
            latent_dim_in=latent_dim + n_conditions,
            channel_in_disc=n_channels + n_conditions,
            n_channels=n_channels,
            n_conditions=n_conditions,
            sequence_length_generated=sequence_length,
            device=device,
            hidden_dim=state_dict['configuration']['hidden_dim'],
            num_layers=state_dict['configuration']['num_layers'],
            patch_size=state_dict['configuration']['patch_size'],
            autoencoder=state_dict['configuration']['autoencoder'],
        )
        generator.load_state_dict(state_dict['generator'])
        generator.to(device).eval()
        
        # Carichiamo i pesi dell'AE nel discriminatore (per l'encoder)
        ae_checkpoint_path = default_args.get('checkpoint', '')
        
        # 2. Se non l'hai passato nel comando, cercalo nella configurazione del modello GAN
        if ae_checkpoint_path == '':
            if 'checkpoint' in state_dict['configuration']:
                ae_checkpoint_path = state_dict['configuration']['checkpoint']
            elif 'autoencoder' in state_dict['configuration']:
                # A volte la repo salva il percorso sotto 'autoencoder'
                ae_checkpoint_path = state_dict['configuration']['autoencoder']
        
        if ae_checkpoint_path == '' or not os.path.exists(ae_checkpoint_path):
            # Ultimo tentativo: cercalo nella cartella trained_models col nome dell'AE
            # (Adatta il nome se necessario)
            print(f">>> ATTENZIONE: Percorso AE non trovato o non valido: {ae_checkpoint_path}")
            raise ValueError("Devi specificare il percorso dell'Autoencoder pre-trainato usando checkpoint=...")

        print(f"Loading PURE Encoder weights from: {ae_checkpoint_path}")
        ae_state_dict = torch.load(ae_checkpoint_path, map_location='cpu')
        discriminator.load_state_dict(ae_state_dict['discriminator'])
        discriminator.to(device).eval()

        # 3. Attivazione Wrapper (Fondamentale!)
        if isinstance(generator, DecoderGenerator):
            generator.decode_output(True) # Garantisce il passaggio 64 -> 128
        if isinstance(discriminator, EncoderDiscriminator):
            discriminator.encode_input(True) # Garantisce il passaggio 128 -> 64

        # 4. Estrazione TESTA ENCODER (Solo dopo aver caricato i pesi AE!)
        base_disc = discriminator.discriminator if hasattr(discriminator, 'discriminator') else discriminator
        encoder_head = torch.nn.Sequential(
            base_disc[0], base_disc[1], 
            base_disc[2].clshead[0], base_disc[2].clshead[1]
        ).to(device).eval()
        
        print(">>> GARANZIA: Encoder AE (puro) e Generator GAN (rifinito) pronti.")
        # check given conditions that they are numeric
        for i, x in enumerate(condition):
            if x == -1 or x == -2:
                continue
            else:
                try:
                    condition[i] = float(x)
                except ValueError:
                    raise ValueError(f"Condition {x} is not numeric.")

        seq_len = 1  # max(1, input_sequence_length)
        cond_labels = torch.zeros((num_samples_parallel, seq_len, n_conditions)).to(device) + torch.tensor(condition).to(device)
        cond_labels = cond_labels.to(device)

        # generate samples
        num_sequences = num_samples_total // num_samples_parallel
        print("Generating samples...")

        all_samples = np.zeros((num_samples_parallel * num_sequences * n_channels, n_conditions + 1 + sequence_length))

        for i in range(num_sequences):
            print(f"Generating sequence {i + 1}/{num_sequences}...")
            with torch.no_grad():
                # A. Scegliamo trial reali casuali dal set filtrato per il batch corrente
                idx = np.random.choice(len(filtered_real_data), num_samples_parallel)
                batch_real = filtered_real_data[idx].to(device).float()
                
                # B. Prepariamo l'input per l'encoder (Encoding temporale + 9° canale)
                # n_cond è n_conditions
                real_eeg = batch_real[:, n_conditions:, :]
                # Compressione temporale (128 -> 64)
                real_reduced = discriminator.encoder.encode(real_eeg) if discriminator.encode else real_eeg
                
                # Aggiunta canale classe (make_fake_data)
                # Usiamo la classe reale dei trial estratti
                labels_batch = batch_real[:, :n_conditions, 0].unsqueeze(1)
                repeated_l = labels_batch.repeat(1, 1, real_reduced.shape[1]).permute(0, 2, 1)
                encoder_input = torch.cat((real_reduced, repeated_l), dim=-1)

                # C. Estrazione z e PERTURBAZIONE (Eq. 7)
                z_real = encoder_head(encoder_input)
                noise = torch.randn_like(z_real)
                z_perturbed = z_real + (epsilon * noise)

                # D. Concatenazione finale per il Generatore [B, 128 + 1]
                z_final = torch.cat((z_perturbed, labels_batch.view(num_samples_parallel, -1)), dim=-1)
                
                # E. Generazione (restituisce 128 punti grazie al Decoder Temporale)
                samples = generator(z_final).cpu().numpy()

            # Reshape e Formatting
            new_samples = np.zeros((num_samples_parallel * n_channels, n_conditions + 1 + sequence_length))
            for j, channel in enumerate(channel_names):
                # Header: [Classe_Target, Nome_Elettrodo, Dati_EEG...]
                cond_col = np.full((num_samples_parallel, 1), target_val)
                chan_col = np.full((num_samples_parallel, 1), channel)
                new_samples[j::n_channels] = np.concatenate((cond_col, chan_col, samples[:, :, j]), axis=-1)
            
            all_samples[i * num_samples_parallel * n_channels:(i + 1) * num_samples_parallel * n_channels] = new_samples
    elif state_dict['configuration']['model_class'] == 'VariationalAutoencoder':

        # load data
        dataloader = Dataloader(path=state_dict['configuration']['dataloader']['data'],
                        kw_channel=kw_channel[0], 
                        kw_conditions=state_dict['configuration']['dataloader']['kw_conditions'],
                        kw_time=state_dict['configuration']['dataloader']['kw_time'],
                        norm_data=state_dict['configuration']['dataloader']['norm_data'], 
                        std_data=state_dict['configuration']['dataloader']['std_data'], 
                        diff_data=state_dict['configuration']['dataloader']['diff_data'])        
        dataset = dataloader.get_data()
        dataset = DataLoader(dataset, batch_size=state_dict['configuration']['batch_size'], shuffle=True)

        sequence_length = int(state_dict['configuration']['input_dim']/dataset.dataset.shape[-1])
        channel_names = dataloader.channels
        n_conditions = len(default_args['conditions'])
        if condition:
            cond_labels = torch.zeros((num_samples_total, state_dict['configuration']['input_dim'], len(default_args['conditions']))).to(device) + torch.tensor(condition).to(device)
        else:
            cond_labels = torch.zeros((num_samples_total, state_dict['configuration']['input_dim'], 1)).to(device) + torch.tensor([-1]).to(device)
        cond_labels = cond_labels.to(device)

        # load VAE
        model = VariationalAutoencoder(input_dim=state_dict['configuration']['input_dim'], 
                                   hidden_dim=state_dict['configuration']['hidden_dim'], 
                                   encoded_dim=state_dict['configuration']['encoded_dim'], 
                                   activation=state_dict['configuration']['activation'],
                                   device=device).to(device)
        
        consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.')
        model.load_state_dict(state_dict['model'])

        # generate samples
        samples = model.generate_samples(loader=dataset, condition=condition, num_samples=num_samples_total)

        # reconfigure samples to a 2D matrix for saving
        new_samples = []
        for j, channel in enumerate(channel_names):
            new_samples.append(np.concatenate((cond_labels.cpu().numpy()[:, 0, :], np.zeros((num_samples_total, 1)) + channel, samples[:, 1:, j]), axis=-1))
        # add samples to all_samples
        all_samples = np.vstack(new_samples)
    
    else:
        raise NotImplementedError(f"The model class {state_dict['configuration']['model_class']} is not recognized.")

    # save samples
    print("Saving samples...")

    # create time step labels
    time_labels = [f'{kw_time}{i}' for i in range(sequence_length)]
    # create dataframe
    df = pd.DataFrame(all_samples, columns=[col_labels + kw_channel + time_labels])
    df.to_csv(path_samples, index=False)

    print("Generated samples were saved to " + path_samples)
        
if __name__ == '__main__':
    main()
