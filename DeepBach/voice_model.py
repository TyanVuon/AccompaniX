"""
@author: Gaetan Hadjeres
"""
import torch.nn.functional as F
import random
import os
import datetime
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from DatasetManager.chorale_dataset import ChoraleDataset
from DeepBach.helpers import cuda_variable, init_hidden

from torch import nn

from DeepBach.data_utils import reverse_tensor, mask_entry


class VoiceModel(nn.Module):
    def __init__(self,
                 dataset: ChoraleDataset,
                 main_voice_index: int,
                 note_embedding_dim: int,
                 meta_embedding_dim: int,
                 num_layers: int,
                 lstm_hidden_size: int,
                 dropout_lstm: float,
                 num_epochs: int,
                 num_notes_per_voice: list,
                 hidden_size_linear=int,
                 batch_size=int,
                 metadata_values=dict
    ):
        #("Type of metadata_values in VoiceModel:", type(metadata_values))
        #print("Contents of metadata_values in VoiceModel:", metadata_values)

    # Reordered hidden_size_linear and num_epochs is 200 supposedly
        super(VoiceModel, self).__init__()
        self.dataset = dataset
        self.main_voice_index = main_voice_index
        self.note_embedding_dim = note_embedding_dim
        self.meta_embedding_dim = meta_embedding_dim
        self.num_notes_per_voice = [len(d)
                                    for d in dataset.note2index_dicts]
        self.num_voices = self.dataset.num_voices
        self.num_metas_per_voice = [
                                       metadata.num_values
                                       for metadata in dataset.metadatas
                                   ] + [self.num_voices]
        self.num_metas = len(self.dataset.metadatas) + 1
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_lstm = dropout_lstm
        self.hidden_size_linear = hidden_size_linear


        self.num_epochs = num_epochs
        self.other_voices_indexes = [i
                                     for i
                                     in range(self.num_voices)
                                     if not i == main_voice_index]

        self.note_embeddings = nn.ModuleList(
            [nn.Embedding(num_notes, note_embedding_dim)
             for num_notes in self.num_notes_per_voice]
        )
        self.meta_embeddings = nn.ModuleList(
            [nn.Embedding(num_metas, meta_embedding_dim)
             for num_metas in self.num_metas_per_voice]
        )


        self.lstm_left = nn.LSTM(input_size=note_embedding_dim * self.num_voices +
                                            meta_embedding_dim * self.num_metas,
                                 hidden_size=lstm_hidden_size,
                                 num_layers=num_layers,
                                 dropout=dropout_lstm,
                                 batch_first=True)
        self.lstm_right = nn.LSTM(input_size=note_embedding_dim * self.num_voices +
                                             meta_embedding_dim * self.num_metas,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout_lstm,
                                  batch_first=True)

        self.mlp_center = nn.Sequential(
            nn.Linear((note_embedding_dim * (self.num_voices - 1)
                       + meta_embedding_dim * self.num_metas),
                      hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, lstm_hidden_size)
        )


        self.mlp_predictions = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 3,
                      hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, self.num_notes_per_voice[main_voice_index])
        )
        self.batch_size = batch_size

        self.tenor_cache = []  # Initialization of the cache
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #print("Type of metadata_values just before error:", type(metadata_values))
        #print("Contents of metadata_values just before error:", metadata_values)

        self.num_metadata_values = sum(metadata_values.values())
        self.cache_names = ['m3cache', 'm2cache', 'm1cache', 'm05cache', 'p05cache', 'p1cache', 'p2cache', 'p3cache']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move the model to the correct device
        self.num_notes_per_voice = num_notes_per_voice

        cache_size = 16  # Assuming a fixed size for the cache
        for cache_name in self.cache_names:
            random_tensor = torch.randn(cache_size, dtype=torch.float32).to(self.device)
            setattr(self, cache_name, random_tensor)



    def forward(self, *input):
        notes, metas = input
        batch_size, num_voices, timesteps_ticks = notes[0].size()

        # put time first
        ln, cn, rn = notes
        ln, rn = [t.transpose(1, 2)
                  for t in (ln, rn)]
        notes = ln, cn, rn

        # embedding
        notes_embedded = self.embed(notes, type='note')
        metas_embedded = self.embed(metas, type='meta')
        # lists of (N, timesteps_ticks, voices * dim_embedding)
        # where timesteps_ticks is 1 for central parts

        # concat notes and metas
        input_embedded = [torch.cat([notes, metas], 2) if notes is not None else None
                          for notes, metas in zip(notes_embedded, metas_embedded)]

        left, center, right = input_embedded

        # print("Forward - Left Shape:", left.shape,
        #       "Center Shape:", center.shape if center is not None else None,
        #       "Right Shape:", right.shape)

        # # Example: expand dimensions proportionally
        # expansion_factor = 160 / (self.note_embedding_dim * self.num_voices +
        #                           self.meta_embedding_dim * self.num_metas)
        # left = torch.nn.functional.interpolate(left, size=int(left.shape[2] * expansion_factor))
        # right = torch.nn.functional.interpolate(right, size=int(right.shape[2] * expansion_factor))
        # if center is not None:
        #     center = torch.nn.functional.interpolate(center, size=int(center.shape[2] * expansion_factor))

        # main part
        hidden = init_hidden(
            num_layers=self.num_layers,
            batch_size=batch_size,
            lstm_hidden_size=self.lstm_hidden_size,
        )
        left, hidden = self.lstm_left(left, hidden)
        left = left[:, -1, :]

        if self.num_voices == 1:
            center = cuda_variable(torch.zeros(
                batch_size,
                self.lstm_hidden_size)
            )
        else:
            center = center[:, 0, :]  # remove time dimension
            center = self.mlp_center(center)

        hidden = init_hidden(
            num_layers=self.num_layers,
            batch_size=batch_size,
            lstm_hidden_size=self.lstm_hidden_size,
        )
        right, hidden = self.lstm_right(right, hidden)
        right = right[:, -1, :]

        # concat and return prediction
        predictions = torch.cat([
            left, center, right
        ], 1)

        predictions = self.mlp_predictions(predictions)
        ##print("Predictions data type at forward method output:", predictions.dtype)

        return predictions

    def preprocess_input(self, tensor_chorale, tensor_metadata):
        """
        :param tensor_chorale: (batch_size, num_voices, chorale_length_ticks)
        :param tensor_metadata: (batch_size, num_metadata, chorale_length_ticks)
        :return: (notes, metas, label) tuple
        where
        notes = (left_notes, central_notes, right_notes)
        metas = (left_metas, central_metas, right_metas)
        label = (batch_size)
        right_notes and right_metas are REVERSED (from right to left)
        """
        batch_size, num_voices, chorale_length_ticks = tensor_chorale.size()

        # random shift! Depends on the dataset
        offset = random.randint(0, self.dataset.subdivision)
        time_index_ticks = chorale_length_ticks // 2 + offset

        # split notes
        notes, label = self.preprocess_notes(tensor_chorale, time_index_ticks)
        metas = self.preprocess_metas(tensor_metadata, time_index_ticks)

        ##print("Preprocess Input - Notes Shape:", notes[0].shape, "Metas Shape:", metas[0].shape)

        return notes, metas, label

    def preprocess_notes(self, tensor_chorale, time_index_ticks):
        """

        :param tensor_chorale: (batch_size, num_voices, chorale_length_ticks)
        :param time_index_ticks:
        :return:
        """
        batch_size, num_voices, _ = tensor_chorale.size()
        left_notes = tensor_chorale[:, :, :time_index_ticks]
        right_notes = reverse_tensor(
            tensor_chorale[:, :, time_index_ticks + 1:],
            dim=2)
        if self.num_voices == 1:
            central_notes = None
        else:
            central_notes = mask_entry(tensor_chorale[:, :, time_index_ticks],
                                       entry_index=self.main_voice_index,
                                       dim=1)
        label = tensor_chorale[:, self.main_voice_index, time_index_ticks]
        return (left_notes, central_notes, right_notes), label

    def preprocess_metas(self, tensor_metadata, time_index_ticks):
        """

        :param tensor_metadata: (batch_size, num_voices, chorale_length_ticks)
        :param time_index_ticks:
        :return:
        """

        left_metas = tensor_metadata[:, self.main_voice_index, :time_index_ticks, :]
        right_metas = reverse_tensor(
            tensor_metadata[:, self.main_voice_index, time_index_ticks + 1:, :],
            dim=1)
        central_metas = tensor_metadata[:, self.main_voice_index, time_index_ticks, :]
        return left_metas, central_metas, right_metas

    def embed(self, notes_or_metas, type):
        if type == 'note':
            embeddings = self.note_embeddings
            embedding_dim = self.note_embedding_dim
            other_voices_indexes = self.other_voices_indexes
        elif type == 'meta':
            embeddings = self.meta_embeddings
            embedding_dim = self.meta_embedding_dim
            other_voices_indexes = range(self.num_metas)

        batch_size, timesteps_left_ticks, num_voices = notes_or_metas[0].size()
        batch_size, timesteps_right_ticks, num_voices = notes_or_metas[2].size()

        left, center, right = notes_or_metas
        # center has self.num_voices - 1 voices
        left_embedded = torch.cat([
            embeddings[voice_id](left[:, :, voice_id])[:, :, None, :]
            for voice_id in range(num_voices)
        ], 2)
        right_embedded = torch.cat([
            embeddings[voice_id](right[:, :, voice_id])[:, :, None, :]
            for voice_id in range(num_voices)
        ], 2)
        # Handling for center tensor
        if center is not None and center.size(1) > 0:
            center_embedded = torch.cat([
                embeddings[voice_id](center[:, k].unsqueeze(1))
                for k, voice_id in enumerate(other_voices_indexes)
            ], 1)
            center_embedded = center_embedded.view(batch_size, 1, len(other_voices_indexes) * embedding_dim)
        elif self.num_voices > 1:  # Ensure center_embedded is created if there are more than one voice
            # Return a tensor of zeros with expected dimensions
            center_embedded = torch.zeros(batch_size, 1, (self.num_voices - 1) * embedding_dim, device=center.device)
        else:
            center_embedded = None

        # print("Embed - Left Shape:", left_embedded.shape,
        #      "Center Shape:", center_embedded.shape if center_embedded is not None else None,
        #      "Right Shape:", right_embedded.shape)

        # squeeze two last dimensions for left and right
        left_embedded = left_embedded.view(batch_size, timesteps_left_ticks, num_voices * embedding_dim)
        right_embedded = right_embedded.view(batch_size, timesteps_right_ticks, num_voices * embedding_dim)

        # print("after squeeze Embed - Left Shape:", left_embedded.shape, "Center Shape:",
        #       center_embedded.shape if center_embedded is not None else None, "Right Shape:", right_embedded.shape)
        #

        return left_embedded, center_embedded, right_embedded

    def extract_pitch_differences_from_tensor(self, tensor_voice):
        """
        Extract pitch differences from a tensor representing multiple voices.
        :param tensor_voice: A tensor representing the pitches of voices (batch, time_step).
        :return: A tensor of pitch differences between consecutive notes for each voice in the batch.
        """
        # Assuming tensor_voice shape is (batch_size, time_steps)
        pitch_diffs = torch.diff(tensor_voice, dim=1)
        return pitch_diffs

        ## Pitch difference analysis methods

    def update_cache(self, tensor_chorale):
        if tensor_chorale.ndim != 2:
            raise ValueError("Expected tensor_chorale to be 2-dimensional")

        tenor_voice = tensor_chorale[:, self.main_voice_index]
        pitch_diffs = torch.diff(tenor_voice, dim=0)

        # Cache the first one bar of pitch differences
        subdivision = self.dataset.subdivision
        bar_limit = subdivision * 1  # Adjust as needed for the number of bars
        pitch_diffs = pitch_diffs[:bar_limit]

        # Pad the pitch differences to ensure a uniform length of 16
        if len(pitch_diffs) < 16:
            padding = 16 - len(pitch_diffs)
            pitch_diffs = F.pad(pitch_diffs, (0, padding), "constant", 0)

        # Update the expanded pitch differences cache
        self.expand_pitch_differences(pitch_diffs)

    def analyze_predicted_pitch_diffs(self, predictions):
        """
        Analyze the predicted pitch differences for all voices.

        :param predictions: The output predictions from the model.
        :return: Analyzed or processed pitch differences.
        """
        # Extract pitch differences from predictions for all voices
        predicted_pitch_diffs = self.extract_pitch_differences_from_tensor(predictions)

        # Additional analysis can be added here if needed
        # For example, comparing predicted pitch differences with a reference or cache

        return predicted_pitch_diffs

    def calculate_matching_incentive(self, generated_voices):
        ##print("Shape of generated_voices in calculate_matching_incentive:", generated_voices.shape)

        # Calculate pitch differences for generated voices
        pitch_diffs = torch.diff(generated_voices, dim=1)

        total_incentive = 0

        # Iterate over each cache
        for cache_name in self.cache_names:
            cached_diffs = getattr(self, cache_name)

            # Iterate through pitch_diffs in 16-element windows
            for i in range(pitch_diffs.shape[1] - 15):
                # Extract a 16-element slice from pitch_diffs
                pitch_diff_slice = pitch_diffs[:, i:i + 16]

                # Compare pitch_diff_slice with cached_diffs
                # Calculate the sign agreement
                sign_agreement = torch.sign(pitch_diff_slice) == torch.sign(cached_diffs)
                similarity = sign_agreement.float().mean()  # Calculate the mean of sign agreements as similarity

                # Update total_incentive based on the comparison
                total_incentive += similarity

        return total_incentive

    def calculate_mismatch_penalty(self, predictions):
        """
        Calculate a penalty for mismatch between predicted pitch differences and expected patterns.
        :param predictions: The output predictions from the model.
        :return: Calculated mismatch penalty.
        """
        # Extract pitch differences from predictions
        pitch_diffs = self.extract_pitch_differences_from_tensor(predictions)

        # Retrieve cached pitch differences for each variation
        cache_names = ['m3cache', 'm2cache', 'm1cache', 'm05cache', 'p05cache', 'p1cache', 'p2cache', 'p3cache']
        cached_diffs_tensors = [getattr(self, name) for name in cache_names if getattr(self, name) is not None]

        penalty = 0
        for diff in pitch_diffs:
            mismatch = True
            for cached_diffs in cached_diffs_tensors:
                # Ensure the shapes are compatible for comparison
                if diff.unsqueeze(1).shape[1] == cached_diffs.shape[0]:
                    comparison_result = (diff.unsqueeze(1) == cached_diffs)
                    if torch.any(comparison_result):
                        mismatch = False
                        break
            if mismatch:
                penalty += 0.1

        return penalty

    def modified_loss_function(self, predictions, label, pitch_diff_labels, penalty_weight=0.01,
                               incentive_weight=0.005,
                               pitch_diff_loss_weight=0.1):
        """
        Custom loss function that combines classification loss with custom penalty, incentives, and pitch difference loss.
        """
        ##print("Predictions data type at modified lf input:", predictions.dtype)
        ##print("Label data type at modified lf input:", label.dtype)

        # Standard classification loss (e.g., Cross-Entropy)
        classification_loss = nn.CrossEntropyLoss()(predictions, label)

        # Mismatch penalty based on pitch differences
        mismatch_penalty = self.calculate_mismatch_penalty(predictions)

        # Pitch difference loss calculation
        if pitch_diff_labels is not None:
            predicted_pitch_diffs = self.extract_pitch_differences_from_tensor(predictions)
            pitch_diff_loss = self.calculate_pitch_diff_loss(predicted_pitch_diffs, pitch_diff_labels)
        else:
            pitch_diff_loss = 0

        # Calculate matching incentive
        matching_incentive = self.calculate_matching_incentive(predictions)

        # Combine the losses and incentives                                                            #pitch diff loss is causing RuntimeError: Found dtype Long but expected Float
        total_loss = classification_loss + penalty_weight * mismatch_penalty + pitch_diff_loss_weight * pitch_diff_loss - incentive_weight * matching_incentive
        return total_loss




    def extract_and_compare_pitch_differences(self, tensor_chorale):
        # Extract pitch differences for all voices
        pitch_diffs = [self.extract_pitch_differences(tensor_chorale[:, voice, :]) for voice in
                       range(self.num_voices)]

        # Expand the tenor's pitch differences
        tenor_diffs_expanded = self.expand_pitch_differences(pitch_diffs[self.main_voice_index])

        # Compare other voices' pitch differences to the expanded tenor pitch differences
        penalties = [self.compare_to_expanded_tenor(pitch_diffs[voice], tenor_diffs_expanded)
                     for voice in range(self.num_voices) if voice != self.main_voice_index]

        # Calculate the total penalty
        total_penalty = sum(penalties)

        return total_penalty

    def expand_pitch_differences(self, tenor_diffs):
        multipliers = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
        cache_names = ['m3cache', 'm2cache', 'm1cache', 'm05cache', 'p05cache', 'p1cache', 'p2cache', 'p3cache']
        for multiplier, cache_name in zip(multipliers, cache_names):
            expanded_diffs = tenor_diffs * multiplier
            setattr(self, cache_name, expanded_diffs)  # Save each variation in a separate attribute

    def compare_to_expanded_tenor(self, voice_diffs, expanded_tenor_diffs):
        penalty = 0
        for diff in voice_diffs:
            if diff not in expanded_tenor_diffs:
                penalty += 0.1
        return penalty

        ## Loss Calculation Methods


    def calculate_pitch_diff_loss(self, predicted_diffs, true_diffs):
        mse_loss = nn.MSELoss()
        total_loss = 0

        ## Conversion here is important for custom loss function to integrate properly
        predicted_diffs = predicted_diffs.float()
        true_diffs = true_diffs.float()


        # Ensure true_diffs and predicted_diffs are the same length
        min_length = min(predicted_diffs.size(1), true_diffs.size(1))
        predicted_diffs = predicted_diffs[:, :min_length]
        true_diffs = true_diffs[:, :min_length]



        # Reshape true_diffs to match predicted_diffs if necessary
        if true_diffs.dim() == 3 and true_diffs.size(2) != predicted_diffs.size(1):
            true_diffs = true_diffs.view(true_diffs.size(0), -1)

        # Iterate over predicted_diffs in steps of 31, wrapping around
        for start in range(0, min_length, 31):
            end = min(start + 31, min_length)
            predicted_segment = predicted_diffs[:, start:end]

            # Calculate and sum up the losses
            segment_loss = mse_loss(predicted_segment, true_diffs[:, start:end])
            total_loss += segment_loss

        return total_loss






    def save(self, details):
        # Ensure 'details' contains all required keys
        required_keys = ['num_epochs', 'batch_size', 'num_iterations', 'lstm_hidden_size', 'note_embedding_dim',
                         'meta_embedding_dim', 'num_layers', 'dropout_lstm', 'linear_hidden_size',
                         'sequence_length_ticks']
        if not all(key in details for key in required_keys):
            print("Error: 'details' is missing required keys.")
            print("Missing keys:", [key for key in required_keys if key not in details])
            return

        # Construct filename using values from 'details'
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"model_{timestamp}_vi{self.main_voice_index}_ep{details['num_epochs']}_bs{details['batch_size']}_ni{details['num_iterations']}_lhs{details['lstm_hidden_size']}_ned{details['note_embedding_dim']}_med{details['meta_embedding_dim']}_nl{details['num_layers']}_dl{details['dropout_lstm']}_lh{details['linear_hidden_size']}_slt{details['sequence_length_ticks']}.pt"

        # Ensure the directory exists
        save_dir = 'models'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        # Prepare the model state and num_notes_per_voice for saving
        save_state = {
            'state_dict': self.state_dict(),
            'num_notes_per_voice': self.num_notes_per_voice  # Include num_notes_per_voice in the saved state
        }
        torch.save(save_state, save_path)


    def __repr__(self):
        return f'model_Dec09th17h_vi{self.main_voice_index}_ep{self.num_epochs}_bs{self.batch_size}_ni{self.note_embedding_dim}_lhs{self.lstm_hidden_size}_ned{self.num_layers}_med{self.meta_embedding_dim}'

    # Save the state dictionary at the end of training
    def save_state_dict(self, file_path):
        torch.save(self.state_dict(), file_path)

    def train_model(self, batch_size=16, num_epochs=10, optimizer=None, details=None):

        # Print the loaded model file name at the start of training
        loaded_model_file = getattr(self, 'loaded_model_file', 'Unknown')
        print(f"Training model loaded from file: {loaded_model_file}")

        if details is None:
            print("Warning: 'details' dictionary not provided. Model will not be saved.")
        else:
            print(f"Training Details: {details}")

        (dataloader_train, dataloader_val, _) = self.dataset.data_loaders(batch_size=batch_size)
        total_steps = self.num_voices * num_epochs * len(self.dataset.data_loaders(batch_size=batch_size)[0])
        progress_bar = tqdm(total=total_steps, desc="Overall Training Progress")

        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            print(f'===Epoch {epoch}/{num_epochs}===')

            # Training phase On
            self.train()  # Switch to training mode

            # Initialize the progress bar for the current epoch
            with tqdm(total=len(dataloader_train), desc=f"Epoch {epoch}/{num_epochs} Batch Progress") as epoch_progress:
                # Training phase
                for batch_idx, (tensor_chorale, tensor_metadata) in enumerate(dataloader_train):
                    ##print(f'Processing batch {batch_idx + 1}/{len(dataloader_train)} in training')
                    tensor_chorale = cuda_variable(tensor_chorale).long()
                    tensor_metadata = cuda_variable(tensor_metadata).long()

                    # Get processed inputs
                    notes, metas, label = self.preprocess_input(tensor_chorale, tensor_metadata)
                    # If pitch_diff_labels are part of your model, uncomment the next line
                    pitch_diff_labels = self.extract_pitch_differences_from_tensor(tensor_chorale)

                    # Forward pass
                    predictions = self.forward(notes, metas)
                    ##print("Predictions data type before loss:", predictions.dtype)

                    # Ensure label is of type torch.long
                    label = label.long()
                    ##print("Label data type before loss:", label.dtype)

                    # Calculate loss
                    # If pitch_diff_labels are used, replace 'label' with 'pitch_diff_labels' in the next line
                    loss = self.modified_loss_function(predictions, label, pitch_diff_labels) if hasattr(self,'modified_loss_function') \
                        else nn.CrossEntropyLoss()(predictions, label)
                    ##print(f'Batch {batch_idx + 1}: Training loss calculated.')
                    ##print("Loss data type before backward:", loss.dtype)

                    # Backpropagation
                    if optimizer:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        ##print(f'Batch {batch_idx + 1}: Backpropagation completed.')
                    # Update the progress bar after each batch
                    epoch_progress.update(1)
                # Calculate training loss and accuracy
                train_loss, train_acc = self.loss_and_acc(dataloader_train, optimizer=optimizer, phase='train')
                print(f'Epoch {epoch}: Training loss = {train_loss}, Accuracy = {train_acc}%')


            # Validation phase On
            self.eval()  # Switch to evaluation mode

            # Validation phase
            val_loss, val_acc = self.loss_and_acc(dataloader_val, optimizer=None, phase='test')
            print(f'Epoch {epoch}: Validation loss = {val_loss}, Accuracy = {val_acc}%')

            # Save the model at the end of each epoch if details are provided
            if details is not None:
                self.save(details)


        print("Training completed.")

        # At the end of training, save the state dictionary
        if details is not None:
            save_file_path = os.path.join('models', f'model_state_dict_{details["num_epochs"]}_epochs.pt')
            self.save_state_dict(save_file_path)

    def loss_and_acc(self, dataloader,
                     optimizer=None,
                     phase='train'):

        average_loss = 0
        average_acc = 0
        if phase == 'train':
            self.train()
        elif phase == 'eval' or phase == 'test':
            self.eval()
        else:
            raise NotImplementedError
        for tensor_chorale, tensor_metadata in dataloader:

            # to Variable
            tensor_chorale = cuda_variable(tensor_chorale).long()
            tensor_metadata = cuda_variable(tensor_metadata).long()

            # preprocessing to put in the DeepBach format
            # see Fig. 4 in DeepBach paper:
            # https://arxiv.org/pdf/1612.01010.pdf
            notes, metas, label = self.preprocess_input(tensor_chorale,
                                                        tensor_metadata)

            weights = self.forward(notes, metas)

            loss_function = torch.nn.CrossEntropyLoss()

            loss = loss_function(weights, label)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = self.accuracy(weights=weights,
                                target=label)

            average_loss += loss.item()
            average_acc += acc.item()

        average_loss /= len(dataloader)
        average_acc /= len(dataloader)
        return average_loss, average_acc

    def accuracy(self, weights, target):
        batch_size = target.size()
        softmax = nn.Softmax(dim=1)(weights)
        pred = softmax.max(1)[1].type_as(target)
        num_corrects = (pred == target).float().sum()
        return num_corrects / batch_size * 100




    def plot_pitch_differences(self, pitch_diffs, bar_limit):
        """
        Plot the pitch differences for the tenor voice of the first bar.
        :param pitch_diffs: List of pitch differences.
        :param bar_limit: The number of differences to plot (e.g., subdivision for one bar).
        """
        # Ensure we only plot the pitch differences up to the bar limit
        pitch_diffs_to_plot = pitch_diffs[:bar_limit]

        plt.figure(figsize=(10, 4))
        plt.plot(pitch_diffs_to_plot, marker='o')
        plt.title('Pitch Differences for the Tenor Voice of the First Bar')
        plt.xlabel('Note Index')
        plt.ylabel('Pitch Difference')
        plt.grid(True)
        plt.show()

    def extract_pitch_curves(self, tenor_sequence):
        # Assuming tenor_sequence is a tensor containing the MIDI values for the tenor voice
            pitch_diffs = [tenor_sequence[i + 1] - tenor_sequence[i] for i in range(len(tenor_sequence) - 1)]
            return pitch_diffs
