"""
@author: Gaetan Hadjeres
"""

import click
import torch
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from DeepBach.model_manager import DeepBach



@click.command()
@click.option('--note_embedding_dim', default=20,
              help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20,
              help='size of the metadata embeddings')
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256,
              help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5,
              help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256,
              help='hidden size of the Linear layers')
@click.option('--batch_size', default=256,
              help='training '
                   'batch size')
@click.option('--num_epochs', default=5,
              help='number of training epochs')
@click.option('--train', is_flag=True,
              help='train the specified model for num_epochs')
@click.option('--num_iterations', default=500,
              help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=64,
              help='length of the generated chorale (in ticks)')
@click.option('--load', is_flag=True, help='Load a model')
@click.option('--batch_size', type=int, help='Model batch size')
@click.option('--num_epochs', type=int, help='Model number of epochs')
@click.option('--num_iterations', type=int, default='100', help='Model number of iterations')
@click.option('--timestamp', type=str, help='Model timestamp')
@click.option('--weights_paths', default='',
              help='Comma-separated paths to the weight files for each voice model')


def main(note_embedding_dim,
         meta_embedding_dim,
         num_layers,
         lstm_hidden_size,
         dropout_lstm,
         linear_hidden_size,
         batch_size,
         num_epochs,
         train,
         num_iterations,
         sequence_length_ticks,
         load,
         timestamp,
         weights_paths
         ):

    dataset_manager = DatasetManager()

    metadatas = [
       FermataMetadata(),
       TickMetadata(subdivision=4),
       KeyMetadata()
    ]
    chorale_dataset_kwargs = {
        'voice_ids':      [0, 1, 2, 3],
        'metadatas':      metadatas,
        'sequences_size': 8,
        'subdivision':    4
    }
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales',
        **chorale_dataset_kwargs
        )
    dataset = bach_chorales_dataset

    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=note_embedding_dim,
        meta_embedding_dim=meta_embedding_dim,
        num_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size,
        dropout_lstm=dropout_lstm,
        linear_hidden_size=linear_hidden_size
    )

    # Inside main() function
    if train:
        ## Load weights if provided
        if weights_paths:
            weights_paths_list = weights_paths.split(',')
            if len(weights_paths_list) != 4:
                raise ValueError("Expected 4 weight files for the 4 voice models")
            for i, weight_path in enumerate(weights_paths_list):
                deepbach.voice_models[i].load_state_dict(torch.load(weight_path))
                print(f"Loaded weights for voice {i} from {weight_path}")

        deepbach.train(batch_size=batch_size, num_epochs=num_epochs)
    elif load:
        if num_iterations == 'None':
            num_iterations = None
        else:
            num_iterations = int(num_iterations)
        deepbach.load(batch_size=batch_size, num_epochs=num_epochs, num_iterations=num_iterations, timestamp=timestamp)
        deepbach.cuda()

    print('Generation')
    if weights_paths:
        weights_paths_list = weights_paths.split(',')
        if len(weights_paths_list) != 4:
            raise ValueError("Expected 4 weight files for the 4 voice models")
        for i, weight_path in enumerate(weights_paths_list):
            deepbach.voice_models[i].load_state_dict(torch.load(weight_path))
            print(f"Loaded weights for voice {i} from {weight_path}")

    score, tensor_chorale, tensor_metadata = deepbach.generation(
        num_iterations=num_iterations,
        sequence_length_ticks=sequence_length_ticks,
    )
    score.show('txt')
    score.show()


if __name__ == '__main__':
    main()
