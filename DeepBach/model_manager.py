"""
@author: Gaetan Hadjeres
"""

from DatasetManager.metadata import FermataMetadata
import numpy as np
import torch
from DeepBach.helpers import cuda_variable, to_numpy
import glob
import os
from torch import optim, nn
from tqdm import tqdm
import time
import datetime
from DeepBach.voice_model import VoiceModel


class DeepBach:
    def __init__(self,
                 dataset,
                 note_embedding_dim,
                 meta_embedding_dim,
                 num_layers,
                 lstm_hidden_size,
                 dropout_lstm,
                 linear_hidden_size,
                 num_epochs,
                 batch_size
                 ):
        self.dataset = dataset
        self.note_embedding_dim = note_embedding_dim
        self.meta_embedding_dim = meta_embedding_dim
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_lstm = dropout_lstm
        self.linear_hidden_size = linear_hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_voices = self.dataset.num_voices
        self.num_metas = len(self.dataset.metadatas) + 1
        self.activate_cuda = torch.cuda.is_available()
        # Calculate num_notes_per_voice and set it as an attribute
        self.num_notes_per_voice = [len(d) for d in self.dataset.note2index_dicts]

        # Ensure that metadata_values is initialized in the dataset
        if not hasattr(self.dataset, 'metadata_values'):
            self.dataset.initialize_metadata_values(self.dataset.metadatas, self.dataset.subdivision)

        # Calculate num_notes_per_voice
        num_notes_per_voice = [len(d) for d in self.dataset.note2index_dicts]
        #print(type(self.dataset.metadata_values), self.dataset.metadata_values)
        #print("Type of self.dataset.metadata_values:", type(self.dataset.metadata_values))
        #print("Contents of self.dataset.metadata_values:", self.dataset.metadata_values)

        self.voice_models = [VoiceModel(
            dataset=self.dataset,
            main_voice_index=main_voice_index,
            note_embedding_dim=note_embedding_dim,
            meta_embedding_dim=meta_embedding_dim,
            num_layers=num_layers,
            lstm_hidden_size=lstm_hidden_size,
            dropout_lstm=dropout_lstm,
            num_epochs=num_epochs,
            hidden_size_linear=linear_hidden_size,
            batch_size=batch_size,
            metadata_values=self.dataset.metadata_values,
            num_notes_per_voice=num_notes_per_voice  # Pass the calculated num_notes_per_voice
        )
            for main_voice_index in range(self.num_voices)
        ]

    def cuda(self, main_voice_index=None):
        if self.activate_cuda:
            if main_voice_index is None:
                for voice_index in range(self.num_voices):
                    self.cuda(voice_index)
            else:
                self.voice_models[main_voice_index].cuda()

    def extract_params_from_filename(self, filename, exclude_voice_id=False):
        """
        Extract configuration parameters from the model filename.
        Optionally exclude voice id parameter based on the exclude_voice_id flag.
        """
        # Separate base name from extension
        base_name, _ = os.path.splitext(filename)
        parts = base_name.split('_')
        params = {}
        for part in parts:
            if exclude_voice_id and part.startswith('vi'):
                continue  # Skip voice index part if exclude_voice_id is True
            if part.startswith('ned'):
                params['note_embedding_dim'] = int(part[3:].split('.')[0])
            elif part.startswith('med'):
                params['meta_embedding_dim'] = int(part[3:].split('.')[0])
            elif part.startswith('nl'):
                params['num_layers'] = int(part[2:].split('.')[0])
            elif part.startswith('lhs'):
                params['lstm_hidden_size'] = int(part[3:].split('.')[0])
            elif part.startswith('dl'):
                params['dropout_lstm'] = float(part[2:].split('.')[0])
            elif part.startswith('lh'):
                params['linear_hidden_size'] = int(part[2:].split('.')[0])
        return params

    def configure_voice_models(self, note_embedding_dim=None, meta_embedding_dim=None, num_layers=None, lstm_hidden_size=None, dropout_lstm=None, linear_hidden_size=None,num_epochs=None):
        """
        Dynamically configure the voice models architecture based on the given parameters.
        """
        # Print existing architecture
        print("Current architecture before configuration:")
        for i, model in enumerate(self.voice_models):
            print(f"Voice model {i} architecture: {model}")

        #print("Type of metadata_values before VoiceModel instances:", type(self.dataset.metadata_values))
        #print("Contents of metadata_values before VoiceModel instances:", self.dataset.metadata_values)


        # Use provided values or set default values based on click.options
        num_epochs = num_epochs if num_epochs is not None else 5  # Set a default value for num_epochs
        note_embedding_dim = note_embedding_dim if note_embedding_dim is not None else 20  # Default from click.option
        meta_embedding_dim = meta_embedding_dim if meta_embedding_dim is not None else 20  # Default from click.option
        num_layers = num_layers if num_layers is not None else 2  # Default from click.option
        lstm_hidden_size = lstm_hidden_size if lstm_hidden_size is not None else 256 # Default from click.option
        dropout_lstm = dropout_lstm if dropout_lstm is not None else 0.5  # Default from click.option
        linear_hidden_size = linear_hidden_size if linear_hidden_size is not None else 256  # Default from click.option

        # Create new voice models with the configured parameters
        self.voice_models = [VoiceModel(
            dataset=self.dataset,
            main_voice_index=voice_index,
            note_embedding_dim=note_embedding_dim,
            meta_embedding_dim=meta_embedding_dim,
            num_layers=num_layers,
            lstm_hidden_size=lstm_hidden_size,
            dropout_lstm=dropout_lstm,
            hidden_size_linear=linear_hidden_size,
            num_epochs=num_epochs,
            metadata_values=self.dataset.metadata_values,
            num_notes_per_voice=self.num_notes_per_voice
        ) for voice_index in range(self.num_voices)]


        # Debug: Print type and contents after reinitialization
        #print("Type of metadata_values after reinitialization in configure method outlet:", type(self.dataset.metadata_values))
        #print("Contents of metadata_values after reinitialization in configure method outlet:", self.dataset.metadata_values)

        # Print new architecture
        print("New architecture after configuration:")
        for i, model in enumerate(self.voice_models):
            print(f"Voice model {i} architecture: {model}")


    def extract_voice_index(self, filename):
        # Split the filename and find the part that starts with 'vi'
        parts = filename.split('_')
        for part in parts:
            if part.startswith('vi'):
                # Extract the voice index number and return it as an integer
                return int(part[2:])
        # Return a default value (like 0) if no voice index is found
        return 0

    # Utils
    def load_models(self, search_params=None):
        if search_params:
            # Dynamic loading based on search_params
            pattern = "model_*"
            for param, value in search_params.items():
                if 'vi' not in param:  # Exclude voice id from the search pattern
                    pattern += f"_{param}{value}"
            pattern += "*.pt"

            matching_files = glob.glob(os.path.join('models', pattern))
            sets = {}
            for file in matching_files:
                params = self.extract_params_from_filename(file, exclude_voice_id=True)
                set_identifier = '_'.join([f"{param}{value}" for param, value in params.items()])
                sets.setdefault(set_identifier, []).append(file)

            if sets:
                for set_id, files in sets.items():
                    print(f"Matching files for set {set_id}: {files}")
                    for voice_index in range(self.num_voices):  # Load for each voice
                        voice_files = [f for f in files if f'vi{voice_index}' in f]
                        if not voice_files:  # Handle case where no voice_id in filename
                            voice_files = files  # Use the same files for each voice

                        if voice_files:
                            file_to_load = voice_files[0]  # Load the first matching file
                            print(f"Loading file for voice {voice_index}: {file_to_load}")
                            params = self.extract_params_from_filename(file_to_load)

                            self.configure_voice_models(**params)

                            # Loading the state dictionary directly from the file
                            loaded_data = torch.load(file_to_load)
                            state_dict = loaded_data['state_dict']
                            self.voice_models[voice_index].load_state_dict(state_dict)
                            self.voice_models[voice_index].loaded_model_file = file_to_load
            else:
                print("No matching models found.")
        else:
            # Load default models
            model_files = sorted(glob.glob('models/*.pt'))  # List and sort all .pt files in models folder
            if len(model_files) < self.num_voices:
                print(
                    f"Not enough model files found in 'models/' directory. Expected {self.num_voices}, found {len(model_files)}.")
                return

            for voice_index in range(self.num_voices):
                model_file = model_files[voice_index]
                print(f"Loading model from file: {model_file}")

                # Load the saved model state and num_notes_per_voice
                saved_model = torch.load(model_file)

                # Reinitialize the VoiceModel with the instance variables
                self.voice_models[voice_index] = VoiceModel(
                    dataset=self.dataset,
                    main_voice_index=voice_index,
                    note_embedding_dim=self.note_embedding_dim,
                    meta_embedding_dim=self.meta_embedding_dim,
                    num_layers=self.num_layers,
                    lstm_hidden_size=self.lstm_hidden_size,
                    dropout_lstm=self.dropout_lstm,
                    num_epochs=self.num_epochs,
                    hidden_size_linear=self.linear_hidden_size,
                    batch_size=self.batch_size,
                    metadata_values=self.dataset.metadata_values,
                    num_notes_per_voice=self.num_notes_per_voice
                )

                # Loading the state dictionary directly from the file
                state_dict = saved_model['state_dict']
                self.voice_models[voice_index].load_state_dict(state_dict)
                self.voice_models[voice_index].loaded_model_file = model_file

    # # Example u
# search_params = {'ep': 1, 'ni': 1, 'ned': 20}  # Add more parameters as needed
# deepbach_model = DeepBach(...)
# deepbach_model.load_models(search_params=search_params)

    def save(self, main_voice_index=None, details=None):
        if main_voice_index is None:
            for voice_index in range(self.num_voices):
                self.save(main_voice_index=voice_index, details=details)
        else:
            self.voice_models[main_voice_index].save(details)


    # def save(self, main_voice_index=None):
    #     if main_voice_index is None:
    #         for voice_index in range(self.num_voices):
    #             self.save(main_voice_index=voice_index)
    #     else:
    #         self.voice_models[main_voice_index].save()

    def train(self, batch_size, num_epochs, details=None):
        """
        Train each voice model.

        :param batch_size: Batch size for training.
        :param num_epochs: Number of epochs to train for.
        :param details: Dictionary containing training details and parameters.
        """
        self.train_phase()  # Ensure all models are in training mode

        for voice_index in range(self.num_voices):
            voice_model = self.voice_models[voice_index]
            if self.activate_cuda:
                voice_model.cuda()
            optimizer = optim.Adam(voice_model.parameters())

            # Train each voice model
            voice_model.train_model(optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs, details=details)

            # Optionally, save the model after training
            if details is not None:
                voice_model.save(details)

    def eval_phase(self):
        for voice_model in self.voice_models:
            voice_model.eval()

    def train_phase(self):
        for voice_model in self.voice_models:
            voice_model.train()

    def generation(self,
                   temperature=1.0,
                   batch_size_per_voice=8,
                   num_iterations=None,
                   sequence_length_ticks=160,
                   tensor_chorale=None,
                   tensor_metadata=None,
                   time_index_range_ticks=None,
                   voice_index_range=None,
                   fermatas=None,
                   random_init=True,
                   details=None
                   ):
        """

        :param temperature:
        :param batch_size_per_voice:
        :param num_iterations:
        :param sequence_length_ticks:
        :param tensor_chorale:
        :param tensor_metadata:
        :param time_index_range_ticks: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between timesteps a and b
        :param voice_index_range: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between voice_index a and b
        :param fermatas: list[Fermata]
        :param random_init: boolean, whether or not to randomly initialize
        the portion of the score on which we apply the pseudo-Gibbs algorithm
        :return: tuple (
        generated_score [music21 Stream object],
        tensor_chorale (num_voices, chorale_length) torch.IntTensor,
        tensor_metadata (num_voices, chorale_length, num_metadata) torch.IntTensor
        )
        """
        self.eval_phase()

        # --Process arguments
        # initialize generated chorale
        # tensor_chorale = self.dataset.empty_chorale(sequence_length_ticks)
        if tensor_chorale is None:
            tensor_chorale = self.dataset.random_score_tensor(
                sequence_length_ticks)
        else:
            sequence_length_ticks = tensor_chorale.size(1)

        # initialize metadata
        if tensor_metadata is None:
            test_chorale = next(self.dataset.corpus_it_gen().__iter__())
            tensor_metadata = self.dataset.get_metadata_tensor(test_chorale)

            if tensor_metadata.size(1) < sequence_length_ticks:
                tensor_metadata = tensor_metadata.repeat(1, sequence_length_ticks // tensor_metadata.size(1) + 1, 1)

            # todo do not work if metadata_length_ticks > sequence_length_ticks
            tensor_metadata = tensor_metadata[:, :sequence_length_ticks, :]
        else:
            tensor_metadata_length = tensor_metadata.size(1)
            assert tensor_metadata_length == sequence_length_ticks

        if fermatas is not None:
            tensor_metadata = self.dataset.set_fermatas(tensor_metadata,
                                                        fermatas)

        # timesteps_ticks is the number of ticks on which we unroll the LSTMs
        # it is also the padding size
        timesteps_ticks = self.dataset.sequences_size * self.dataset.subdivision // 2
        if time_index_range_ticks is None:
            time_index_range_ticks = [timesteps_ticks, sequence_length_ticks + timesteps_ticks]
        else:
            a_ticks, b_ticks = time_index_range_ticks
            assert 0 <= a_ticks < b_ticks <= sequence_length_ticks
            time_index_range_ticks = [a_ticks + timesteps_ticks, b_ticks + timesteps_ticks]

        if voice_index_range is None:
            voice_index_range = [0, self.dataset.num_voices]

        tensor_chorale = self.dataset.extract_score_tensor_with_padding(
            tensor_score=tensor_chorale,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks
        )

        tensor_metadata_padded = self.dataset.extract_metadata_with_padding(
            tensor_metadata=tensor_metadata,
            start_tick=-timesteps_ticks,
            end_tick=sequence_length_ticks + timesteps_ticks
        )

        # randomize regenerated part
        if random_init:
            a, b = time_index_range_ticks
            tensor_chorale[voice_index_range[0]:voice_index_range[1], a:b] = self.dataset.random_score_tensor(
                b - a)[voice_index_range[0]:voice_index_range[1], :]

        tensor_chorale = self.parallel_gibbs(
            tensor_chorale=tensor_chorale,
            tensor_metadata=tensor_metadata_padded,
            num_iterations=num_iterations,
            timesteps_ticks=timesteps_ticks,
            temperature=temperature,
            batch_size_per_voice=batch_size_per_voice,
            time_index_range_ticks=time_index_range_ticks,
            voice_index_range=voice_index_range,
        )

        # get fermata tensor
        for metadata_index, metadata in enumerate(self.dataset.metadatas):
            if isinstance(metadata, FermataMetadata):
                break


        score = self.dataset.tensor_to_score(
            tensor_score=tensor_chorale,
            fermata_tensor=tensor_metadata[:, :, metadata_index])

        # Folder where you want to save the scores
        save_folder = 'GeneratedScores'

        # Ensure the folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Construct the filename from details
        if details is not None:
            filename_parts = [f"{key}{value}" for key, value in details.items()]
            filename = f"model_{'_'.join(filename_parts)}.xml"
        else:
            filename = "generated_score.xml"

        # Full path for the file
        full_save_path = os.path.join(save_folder, filename)

        # Save the score
        try:
            score.write('musicxml', full_save_path)
            print(f"Score saved as {full_save_path}")
        except Exception as e:
            print(f"Error saving score: {e}")

        return score, tensor_chorale, tensor_metadata

        return score, tensor_chorale, tensor_metadata

    def parallel_gibbs(self,
                       tensor_chorale,
                       tensor_metadata,
                       timesteps_ticks,
                       num_iterations=1000,
                       batch_size_per_voice=16,
                       temperature=1.,
                       time_index_range_ticks=None,
                       voice_index_range=None,
                       ):
        """
        Parallel pseudo-Gibbs sampling
        tensor_chorale and tensor_metadata are padded with
        timesteps_ticks START_SYMBOLS before,
        timesteps_ticks END_SYMBOLS after
        :param tensor_chorale: (num_voices, chorale_length) tensor
        :param tensor_metadata: (num_voices, chorale_length) tensor
        :param timesteps_ticks:
        :param num_iterations: number of Gibbs sampling iterations
        :param batch_size_per_voice: number of simultaneous parallel updates
        :param temperature: final temperature after simulated annealing
        :param time_index_range_ticks: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between timesteps a and b
        :param voice_index_range: list of two integers [a, b] or None; can be used \
        to regenerate only the portion of the score between voice_index a and b
        :return: (num_voices, chorale_length) tensor
        """
        start_voice, end_voice = voice_index_range
        # add batch_dimension
        tensor_chorale = tensor_chorale.unsqueeze(0)
        tensor_chorale_no_cuda = tensor_chorale.clone()
        tensor_metadata = tensor_metadata.unsqueeze(0)

        # to variable
        tensor_chorale = cuda_variable(tensor_chorale, volatile=True)
        tensor_metadata = cuda_variable(tensor_metadata, volatile=True)

        min_temperature = temperature
        temperature = 1.1

        # Main loop
        for iteration in tqdm(range(num_iterations)):
            # annealing
            temperature = max(min_temperature, temperature * 0.9993)
            # print(temperature)
            time_indexes_ticks = {}
            probas = {}

            for voice_index in range(start_voice, end_voice):
                batch_notes = []
                batch_metas = []

                time_indexes_ticks[voice_index] = []

                # create batches of inputs
                for batch_index in range(batch_size_per_voice):
                    time_index_ticks = np.random.randint(
                        *time_index_range_ticks)
                    time_indexes_ticks[voice_index].append(time_index_ticks)

                    notes, label = (self.voice_models[voice_index]
                                    .preprocess_notes(
                            tensor_chorale=tensor_chorale[
                                           :, :,
                                           time_index_ticks - timesteps_ticks:
                                           time_index_ticks + timesteps_ticks],
                            time_index_ticks=timesteps_ticks
                        )
                    )
                    metas = self.voice_models[voice_index].preprocess_metas(
                        tensor_metadata=tensor_metadata[
                                        :, :,
                                        time_index_ticks - timesteps_ticks:
                                        time_index_ticks + timesteps_ticks,
                                        :],
                        time_index_ticks=timesteps_ticks
                    )

                    batch_notes.append(notes)
                    batch_metas.append(metas)

                # reshape batches
                batch_notes = list(map(list, zip(*batch_notes)))
                batch_notes = [torch.cat(lcr) if lcr[0] is not None else None
                               for lcr in batch_notes]
                batch_metas = list(map(list, zip(*batch_metas)))
                batch_metas = [torch.cat(lcr)
                               for lcr in batch_metas]

                # make all estimations
                probas[voice_index] = (self.voice_models[voice_index]
                                       .forward(batch_notes, batch_metas)
                                       )
                probas[voice_index] = nn.Softmax(dim=1)(probas[voice_index])

            # update all predictions
            for voice_index in range(start_voice, end_voice):
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    probas_pitch = to_numpy(probas_pitch)

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(
                        np.exp(probas_pitch)) - 1e-7

                    # avoid non-probabilities
                    probas_pitch[probas_pitch < 0] = 0

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    tensor_chorale_no_cuda[
                        0,
                        voice_index,
                        time_indexes_ticks[voice_index][batch_index]
                    ] = int(pitch)

            tensor_chorale = cuda_variable(tensor_chorale_no_cuda.clone(),
                                           volatile=True)

        return tensor_chorale_no_cuda[0, :, timesteps_ticks:-timesteps_ticks]
