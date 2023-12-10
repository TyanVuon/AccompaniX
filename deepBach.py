import click
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from DeepBach.model_manager import DeepBach

@click.command()
@click.option('--note_embedding_dim', default=20, help='size of the note embeddings')
@click.option('--meta_embedding_dim', default=20, help='size of the metadata embeddings')
@click.option('--num_layers', default=2, help='number of layers of the LSTMs')
@click.option('--lstm_hidden_size', default=256, help='hidden size of the LSTMs')
@click.option('--dropout_lstm', default=0.5, help='amount of dropout between LSTM layers')
@click.option('--linear_hidden_size', default=256, help='hidden size of the Linear layers')
@click.option('--batch_size', default=256, help='training batch size')
@click.option('--num_epochs', default=5, help='number of training epochs')
@click.option('--train', is_flag=True, help='train the specified model for num_epochs')
@click.option('--num_iterations', default=500, help='number of parallel pseudo-Gibbs sampling iterations')
@click.option('--sequence_length_ticks', default=64, help='length of the generated chorale (in ticks)')
@click.option('--load', default='', help='Parameters to load models')

def main(note_embedding_dim, meta_embedding_dim, num_layers, lstm_hidden_size, dropout_lstm, linear_hidden_size, batch_size, num_epochs, train, num_iterations, sequence_length_ticks, load):
    dataset_manager = DatasetManager()

    metadatas = [FermataMetadata(), TickMetadata(subdivision=4), KeyMetadata()]
    chorale_dataset_kwargs = {'voice_ids': [0, 1, 2, 3], 'metadatas': metadatas, 'sequences_size': 8, 'subdivision': 4}
    bach_chorales_dataset = dataset_manager.get_dataset(name='bach_chorales', **chorale_dataset_kwargs)
    dataset = bach_chorales_dataset

    deepbach = DeepBach(dataset=dataset, note_embedding_dim=note_embedding_dim, meta_embedding_dim=meta_embedding_dim, num_layers=num_layers, lstm_hidden_size=lstm_hidden_size, dropout_lstm=dropout_lstm, linear_hidden_size=linear_hidden_size, num_epochs=num_epochs, batch_size=batch_size)

    # Define details dictionary
    details = {
        'note_embedding_dim': note_embedding_dim,
        'meta_embedding_dim': meta_embedding_dim,
        'num_layers': num_layers,
        'lstm_hidden_size': lstm_hidden_size,
        'dropout_lstm': dropout_lstm,
        'linear_hidden_size': linear_hidden_size,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'num_iterations': num_iterations,
        'sequence_length_ticks': sequence_length_ticks
    }
    metadata_values = {
        'IsPlayingMetadata': 2,  # IsPlayingMetadata has 2 values (playing or not playing)
        'TickMetadata': None,  # TickMetadata's num_values depends on the subdivision, so it's set dynamically
        'ModeMetadata': 3,  # ModeMetadata has 3 values (major, minor, other)
        'KeyMetadata': 16,  # KeyMetadata has 16 values (number of sharps/flats)
        'FermataMetadata': 2  # FermataMetadata has 2 values (with or without fermata)
    }

    if load:
        load_params = dict(param.split('=') for param in load.split(','))
        deepbach.load_models(load_params)
        # Update details with loaded model parameters
        loaded_model_params = deepbach.extract_params_from_filename(load)
        details.update(loaded_model_params)

    if train:
        deepbach.train(batch_size=batch_size, num_epochs=num_epochs, details=details)
    else:
        deepbach.load_models()

    deepbach.cuda()

    print('Generation')
    score, tensor_chorale, tensor_metadata = deepbach.generation(num_iterations=num_iterations, sequence_length_ticks=sequence_length_ticks, details=details)
    score.show('txt')
    score.show()

if __name__ == '__main__':
    main()
