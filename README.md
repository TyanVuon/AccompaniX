![](spiegel.jpg)
# AccompaniX
This repository is a tentative adaptation of the DeepBach model, addressing the coherence challenges in AI music generation. 
The project builds on the foundational work of DeepBach: a Steerable Model for Bach Chorales Generation.
*DeepBach: a Steerable Model for Bach chorales generation*<br/>
Gaëtan Hadjeres, François Pachet, Frank Nielsen<br/>
*ICML 2017 [arXiv:1612.01010](http://proceedings.mlr.press/v70/hadjeres17a.html)*


python 3.10 together with Pytorch 2.1.0+cu121, music21 7.3.3

For the original Keras version, please checkout the `original_keras` branch.

Examples of music generated by Original DeepBach are available on [this website](https://sites.google.com/site/deepbachexamples/)

## Installation


 To set up AccompaniX, follow these steps:
```
git clone https://github.com/TyanVuon/AccompaniX
cd AccompaniX
conda env create --name deepbach_pytorch -f environment.yml

```
This will create a conda env named `deepbach_pytorch`.

### music21 editor 

You might need to
Open a four-part chorale. Press enter on the server address, a list of computed models should appear. Select and (re)load a model. 
[Configure properly the music editor
 called by music21](http://web.mit.edu/music21/doc/moduleReference/moduleEnvironment.html). On Ubuntu you can eg. use MuseScore:

```shell
sudo apt install musescore
python -c 'import music21; music21.environment.set("musicxmlPath", "/usr/bin/musescore")'
```

For usage on a headless server (no X server), just set it to a dummy command:

```shell
python -c 'import music21; music21.environment.set("musicxmlPath", "/bin/true")'
```

## Usage
```
Usage: deepBach.py [OPTIONS]

Options:
  --note_embedding_dim INTEGER    size of the note embeddings
  --meta_embedding_dim INTEGER    size of the metadata embeddings
  --num_layers INTEGER            number of layers of the LSTMs
  --lstm_hidden_size INTEGER      hidden size of the LSTMs
  --dropout_lstm FLOAT            amount of dropout between LSTM layers
  --linear_hidden_size INTEGER    hidden size of the Linear layers
  --batch_size INTEGER            training batch size
  --num_epochs INTEGER            number of training epochs
  --train                         train or retrain the specified model
  --num_iterations INTEGER        number of parallel pseudo-Gibbs sampling
                                  iterations
  --sequence_length_ticks INTEGER
                                  length of the generated chorale (in ticks)
                                  
  --batch_size,--num_epochs,--num_iterations,--timestamp
                                  elements of the matching model to load
  
  --weights_paths                 path to the weights of the model to load prior to training/loading for generations
  --help                          Show this message and exit.
```

## Examples
You can generate a four-bar chorale with the pretrained model and display it in MuseScore  by 
simply running (music21 setup is requried)
```
python deepBach.py
```

You can train a new model from scratch by adding the `--train` flag.
Added command line options are passed to the model constructor. For instance, to train a model with pretrained weights,
to load a model with pretrained weights, to load a specific model by adding the --paths_weights flag which contains either
of the matching parts in the file name, such as iterations, batches, time and so on. 


## Usage with NONOTO
The command 
```
python flask_server.py
```
starts a Flask server listening on port 5000. You can then use 
[NONOTO](https://github.com/SonyCSLParis/NONOTO) to compose with DeepBach in an interactive way.

This server can also been started using Docker with:
```
docker run -p 5000:5000 -it --rm ghadjeres/deepbach
```
(CPU version), with
or
```
docker run --runtime=nvidia -p 5000:5000 -it --rm ghadjeres/deepbach
```
(GPU version, requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).


## Usage within MuseScore
*Deprecated*

Put `deepBachMuseScore.qml` file in your MuseScore plugins directory, and run
```
python musescore_flask_server.py
```
MuseScore3.5,and 4 can be set by configuration option from music21, and mainly for analysis 
during the course of modifications,for interactive use, the server is used.


### Issues

### Music21 editor not set

```
music21.converter.subConverters.SubConverterException: Cannot find a valid application path for format musicxml. Specify this in your Environment by calling environment.set(None, '/path/to/application')
```

Either set it to MuseScore or similar (on a machine with GUI) to to a dummy command (on a server). See the installation section.

# Cited from


```
@InProceedings{pmlr-v70-hadjeres17a,
  title = 	 {{D}eep{B}ach: a Steerable Model for {B}ach Chorales Generation},
  author = 	 {Ga{\"e}tan Hadjeres and Fran{\c{c}}ois Pachet and Frank Nielsen},
  booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
  pages = 	 {1362--1371},
  year = 	 {2017},
  editor = 	 {Doina Precup and Yee Whye Teh},
  volume = 	 {70},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {International Convention Centre, Sydney, Australia},
  month = 	 {06--11 Aug},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v70/hadjeres17a/hadjeres17a.pdf},
  url = 	 {http://proceedings.mlr.press/v70/hadjeres17a.html},
}
```
