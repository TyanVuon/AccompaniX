"""
@author: Gaetan Hadjeres
"""

import glob
import random
import torch.nn.functional as F
import torch
from DatasetManager.chorale_dataset import ChoraleDataset
from DeepBach.helpers import cuda_variable, init_hidden
from music21 import configure
from torch import nn
import matplotlib.pyplot as plt
import datetime
import os

from DeepBach.data_utils import reverse_tensor, mask_entry

MAIN_VOICE_INDEX = 1  # Replace 0 with your desired index

class VoiceModel(nn.Module):
    def __init__(self,
                 dataset: ChoraleDataset,
                 main_voice_index: int,
                 note_embedding_dim: int,
                 meta_embedding_dim: int,
                 num_layers: int,
                 lstm_hidden_size: int,
                 dropout_lstm: float,
                 hidden_size_linear=200,
                 ):

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
        self.tenor_cache = [] #initialization of the cache
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f'Training on {torch.cuda.get_device_name(device)}')
        else:
            device = torch.device("cpu")
            print('Training on CPU')
        self.device = device
        self.to(self.device)
        # 添加 cache_names 属性
        self.cache_names = ['m3cache', 'm2cache', 'm1cache', 'm05cache', 'p05cache', 'p1cache', 'p2cache', 'p3cache']
        print("Note embeddings initialized.")
        ##print(f'Note embeddings size: {[e.num_embeddings for e in self.note_embeddings]}')
        ##print(f'Meta embeddings size: {[e.num_embeddings for e in self.meta_embeddings]}')

    ## data preprocessing methods (3 as below)
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
        ##print(f'tensor_chorale shape: {tensor_chorale.shape}')
        ##print(f'tensor_metadata shape: {tensor_metadata.shape}')
        batch_size, num_voices, chorale_length_ticks = tensor_chorale.size()

        # random shift! Depends on the dataset
        offset = random.randint(0, self.dataset.subdivision)
        time_index_ticks = chorale_length_ticks // 2 + offset

        # Generate pitch difference labels for the main voice
        pitch_diff_labels = self.extract_pitch_differences_from_tensor(tensor_chorale[:, self.main_voice_index, :])

        # Cache tenor voice pitch differences for use in training
        tenor_voice_tensor = tensor_chorale[self.main_voice_index, :]
        tenor_diffs = self.extract_pitch_differences_from_tensor(tenor_voice_tensor)
        self.update_cache(tenor_diffs)

        # split notes
        notes, label = self.preprocess_notes(tensor_chorale, time_index_ticks)
        metas = self.preprocess_metas(tensor_metadata, time_index_ticks)
        # In the preprocess_input method, after calling preprocess_notes and preprocess_metas
        ##print(f'Notes shape after preprocessing: {notes[0].shape}, {notes[1].shape}, {notes[2].shape}')
        ##print(f'Metas shape after preprocessing: {metas[0].shape}, {metas[1].shape}, {metas[2].shape}')

        # Initialize total_loss as None, it will be calculated later in the training loop
        total_loss = None

        return notes, metas, label, pitch_diff_labels

    def preprocess_notes(self, tensor_chorale, time_index_ticks):
        """
        :param tensor_chorale: (batch_size, num_voices, chorale_length_ticks)
        :param time_index_ticks:
        :return:
        """
        batch_size, num_voices, _ = tensor_chorale.size()
        left_notes = tensor_chorale[:, :, :time_index_ticks].transpose(1, 2)
        right_notes = reverse_tensor(
            tensor_chorale[:, :, time_index_ticks + 1:],
            dim=2).transpose(1, 2)
        if self.num_voices == 1:
            central_notes = None
        else:
            central_notes = mask_entry(tensor_chorale[:, :, time_index_ticks],
                                       entry_index=self.main_voice_index,
                                       dim=1)
        label = tensor_chorale[:, self.main_voice_index, time_index_ticks]
        ##print(f'Left notes shape: {left_notes.shape}')
        ##print(f'Right notes shape: {right_notes.shape}')
        ##print(f'Central notes shape: {central_notes.shape if central_notes is not None else "NA"}')

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
        ##print(f'Left metas shape: {left_metas.shape}')
        ##print(f'Right metas shape: {right_metas.shape}')
        ##print(f'Central metas shape: {central_metas.shape}')

        return left_metas, central_metas, right_metas

     ## Embedding method
    def embed(self, notes_or_metas, type):
        if type == 'note':
            embeddings = self.note_embeddings
            embedding_dim = self.note_embedding_dim
            other_voices_indexes = self.other_voices_indexes
        elif type == 'meta':
            embeddings = self.meta_embeddings
            embedding_dim = self.meta_embedding_dim
            other_voices_indexes = range(self.num_metas)

        left, center, right = notes_or_metas

        # Handling left and right embeddings
        left_embedded = torch.cat([embeddings[voice_id](left[:, :, voice_id])[:, :, None, :]
                                   for voice_id in range(len(embeddings))], 2)
        right_embedded = torch.cat([embeddings[voice_id](right[:, :, voice_id])[:, :, None, :]
                                    for voice_id in range(len(embeddings))], 2)

        # Reshaping embedded tensors
        left_embedded = left_embedded.view(left_embedded.shape[0], -1, embedding_dim * len(embeddings))
        right_embedded = right_embedded.view(right_embedded.shape[0], -1, embedding_dim * len(embeddings))

        # Handling center embeddings
        if center is not None:
            center_embedded = torch.cat([embeddings[voice_id](center[:, k].unsqueeze(1))
                                         for k, voice_id in enumerate(other_voices_indexes)], 1)
            center_embedded = center_embedded.view(center.shape[0], 1, -1)
        else:
            center_embedded = None

        return left_embedded, center_embedded, right_embedded

    ## Forward pass method

    def forward(self, input, tensor_chorale=None):
        print("Input Length:", len(input))
        print("Notes shape:", input[0][0].shape)
        print("Metas shape:", input[1][0].shape)

        notes, metas = input[0], input[1]
        batch_size = notes[0].shape[0]

        # Embedding
        notes_embedded = self.embed(notes, type='note')
        metas_embedded = self.embed(metas, type='meta')

        # Extracting embedded notes
        left_embedded, center_embedded, right_embedded = notes_embedded

        # Concatenating notes and metas
        input_embedded = [torch.cat([notes, metas], 2) if notes is not None else None
                          for notes, metas in zip(notes_embedded, metas_embedded)]

        # Processing each segment: left, center, and right
        left, center, right = input_embedded
        hidden = init_hidden(num_layers=self.num_layers, batch_size=batch_size, lstm_hidden_size=self.lstm_hidden_size)
        left, _ = self.lstm_left(left, hidden)
        left = left[:, -1, :]

        if center is not None:
            center = self.mlp_center(center[:, 0, :])
        else:
            center = cuda_variable(torch.zeros(batch_size, self.lstm_hidden_size))

        right, _ = self.lstm_right(right, hidden)
        right = right[:, -1, :]

        # Concatenating the results
        concatenated = torch.cat([left, center, right], 1)

        predictions = self.mlp_predictions(concatenated)

        return predictions

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

    def extract_and_compare_pitch_differences(self, tensor_chorale):
        # Extract pitch differences for all voices
        pitch_diffs = [self.extract_pitch_differences(tensor_chorale[:, voice, :]) for voice in range(self.num_voices)]

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
                penalty += 1
        return penalty



    ## Loss Calculation Methods
    def modified_loss_function(self, predictions, labels, pitch_diff_labels, penalty_weight=0.1, incentive_weight=0.05,
                               pitch_diff_loss_weight=1.0):
        """
        Custom loss function that combines classification loss with custom penalty, incentives, and pitch difference loss.
        """
        # Standard classification loss (e.g., Cross-Entropy)
        classification_loss = nn.CrossEntropyLoss()(predictions, labels)

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

        # Combine the losses and incentives
        total_loss = classification_loss + penalty_weight * mismatch_penalty + pitch_diff_loss_weight * pitch_diff_loss - incentive_weight * matching_incentive
        return total_loss

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
        cached_diffs_tensors = [getattr(self, name) for name in cache_names]

        penalty = 0
        for diff in pitch_diffs:
            mismatch = True
            for cached_diffs in cached_diffs_tensors:
                if torch.any(diff.unsqueeze(1) == cached_diffs):
                    mismatch = False
                    break
            if mismatch:
                penalty += 1

        return penalty

    def calculate_pitch_diff_loss(self, predicted_diffs, true_diffs):
        # 在 calculate_pitch_diff_loss 中确保数据是浮点型
        predicted_diffs = predicted_diffs.float()
        true_diffs = true_diffs.float()

        """
        Calculate the loss based on the differences between predicted and true pitch differences.

        :param predicted_diffs: Predicted pitch differences from the model.
        :param true_diffs: True pitch differences (ground truth).
        :return: Loss value computed using Mean Squared Error.
        """
        mse_loss = nn.MSELoss()
        total_loss = 0

        # Iterate over predicted_diffs in steps of 31, wrapping around
        for start in range(0, len(predicted_diffs[0]), 31):
            end = start + 31
            predicted_segment = predicted_diffs[:, start:end]
            # Adjust the size of predicted_segment if it's shorter than 31
            if predicted_segment.shape[1] < 31:
                predicted_segment = torch.cat((predicted_segment, predicted_diffs[:, :31 - predicted_segment.shape[1]]),
                                              dim=1)

            # Calculate and sum up the losses
            segment_loss = mse_loss(predicted_segment, true_diffs)
            total_loss += segment_loss

        return total_loss

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

    ## Training loop Methods
    def train_model(self,batch_size, num_epochs, num_iterations, optimizer=None):
        batch_size = 1024
        num_epochs = 1
        print("Starting training...")
        for epoch in range(num_epochs):
            print(f'===Epoch {epoch}===')

            # Load dataloaders
            (dataloader_train, dataloader_val, dataloader_test) = self.dataset.data_loaders(batch_size=batch_size)
            print(f'Training DataLoader loaded with {len(dataloader_train)} batches.')
            print(f'Validation DataLoader loaded with {len(dataloader_val)} batches.')

            # Training phase
            for batch_idx, (tensor_chorale, tensor_metadata) in enumerate(dataloader_train):
                print(f'Processing batch {batch_idx + 1}/{len(dataloader_train)} in training')
                tensor_chorale = cuda_variable(tensor_chorale).long()
                tensor_metadata = cuda_variable(tensor_metadata).long()

                # Get processed inputs
                notes, metas, label, pitch_diff_labels = self.preprocess_input(tensor_chorale, tensor_metadata)

                # Forward pass
                predictions = self.forward(notes, metas)

                # Calculate loss
                loss = self.modified_loss_function(predictions, label, pitch_diff_labels)
                print(f'Batch {batch_idx + 1}: Training loss calculated.')

                # Backpropagation
                if hash == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print(f'Batch {batch_idx + 1}: Backpropagation completed.')

            # Calculate training loss and accuracy
            train_loss, train_acc = self.loss_and_acc(dataloader_train, optimizer=optimizer, phase='train')
            print(f'Epoch {epoch}: Training loss = {train_loss}, Accuracy = {train_acc}%')

            # Validation phase
            for batch_idx, (tensor_chorale, tensor_metadata) in enumerate(dataloader_val):
                print(f'Processing batch {batch_idx + 1}/{len(dataloader_val)} in validation')
                tensor_chorale = cuda_variable(tensor_chorale).long()
                tensor_metadata = cuda_variable(tensor_metadata).long()

                # Get processed inputs
                notes, metas, label, pitch_diff_labels = self.preprocess_input(tensor_chorale, tensor_metadata)

                # Forward pass
                predictions = self.forward(notes, metas)

                # Loss calculation (no backpropagation)
                loss = self.modified_loss_function(predictions, label, pitch_diff_labels)
                print(f'Batch {batch_idx + 1}: Validation loss calculated.')

            # Calculate validation loss and accuracy
            val_loss, val_acc = self.loss_and_acc(dataloader_val, optimizer=None, phase='test')
            print(f'Epoch {epoch}: Validation loss = {val_loss}, Accuracy = {val_acc}%')

            # Save model
            self.save(batch_size, num_epochs, num_iterations)
            print(f'Epoch {epoch} completed and model saved.')

        print("Training completed.")

        ##resemblance_penalty is calculated by the extract_and_compare_pitch_differences method, which you need to ensure is implemented correctly.
        ##resemblance_loss is the scaled resemblance penalty. The scaling factor (0.01 in this example) is used to ensure that the resemblance loss does
        # ##not dominate the total loss. You may need to experiment with this scaling factor to find a balance that works for your model.
        ##total_loss is the sum of the classification loss and the resemblance loss, which is then used for the backward pass and optimization if in the training phase.

    def loss_and_acc(self, dataloader, optimizer=None, phase='train'):
        average_loss = 0
        average_acc = 0
        if phase == 'train':
            self.train()
        elif phase in ['eval', 'test']:
            self.eval()
        else:
            raise NotImplementedError

        for tensor_chorale, tensor_metadata in dataloader:
            tensor_chorale = cuda_variable(tensor_chorale).long()
            tensor_metadata = cuda_variable(tensor_metadata).long()

            # Get the processed inputs
            notes, metas, label, pitch_diff_labels = self.preprocess_input(tensor_chorale, tensor_metadata)
            ##print(f'Batch chorale shape: {tensor_chorale.shape}')
            ##print(f'Batch metadata shape: {tensor_metadata.shape}')

            # Forward pass
            predictions = self.forward(notes, metas)
            # In the train_model method, after getting predictions from the forward pass
            print(f'Predictions shape: {predictions.shape}')

            # Calculate the total loss
            total_loss = self.modified_loss_function(predictions, label, pitch_diff_labels)

            if phase == 'train':
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            acc = self.accuracy(predictions, label)

            average_loss += total_loss.item()
            average_acc += acc.item()

        average_loss /= len(dataloader)
        average_acc /= len(dataloader)
        print(f'Dataloader length: {len(dataloader)}')

        return average_loss, average_acc

    ## Utility methods
    def save(self, batch_size, num_epochs, num_iterations):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'models/VoiceModel_bs{batch_size}_ep{num_epochs}_it{num_iterations}_{timestamp}.pt'
        torch.save(self.state_dict(), filename)
        print(f'Model saved as {filename}')

    def load(self, batch_size=None, num_epochs=None, num_iterations=None, timestamp=None):
        model_directory = 'models'
        matched_files = []
        # Iterate over all files in the model directory
        for filename in os.listdir(model_directory):
            if filename.startswith('VoiceModel'):
                match = True
                if batch_size and f'bs{batch_size}' not in filename:
                    match = False
                if num_epochs and f'ep{num_epochs}' not in filename:
                    match = False
                if num_iterations:
                    if f'it{num_iterations}' not in filename and 'itNone' not in filename:
                        match = False
                if timestamp and timestamp not in filename:
                    match = False

                if match:
                    matched_files.append(filename)

        if matched_files:
            # Load the first matched file
            filepath = os.path.join(model_directory, matched_files[0])
            state_dict = torch.load(filepath, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print(f'Model loaded from {filepath}')
        else:
            print('No matching model found.')

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

    def __repr__(self):
        return f'VoiceModel(' \
                   f'{self.dataset.__repr__()},' \
                   f'{self.main_voice_index},' \
                   f'{self.note_embedding_dim},' \
                   f'{self.meta_embedding_dim},' \
                   f'{self.num_layers},' \
                   f'{self.lstm_hidden_size},' \
                   f'{self.dropout_lstm},' \
                   f'{self.hidden_size_linear}' \
                   f')'

    def accuracy(self, predictions, target):
            """
            Calculate the accuracy of the predictions.
            """
            pred = predictions.max(1)[1].type_as(target)
            num_corrects = (pred == target).float().sum()
            return num_corrects / target.size(0) * 100

