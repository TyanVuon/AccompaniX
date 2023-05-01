import music21

# Path to the directory where the chorales dataset is stored
dataset_path = r"C:\Users\Tyan\DPBCT\DeepBachTyan\DatasetManager\datasets\bach_chorales"

# Load the chorales dataset using music21
chorales_dataset = music21.corpus.corpora.LocalCorpus('bach-chorales')
chorales_dataset.addPath(str(dataset_path))

# Print the number of sources in the dataset
print(len(chorales_dataset.getPaths()))




