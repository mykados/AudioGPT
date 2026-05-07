# AudioGPT

# Latest Autoregression Model

## Dataset

The dataset of audio was a personal collection of audio files, to avoid any issues with the distribution of media, the dataset will not be public.

## Models

The models are fairly large around ~4-6GB and will also not be included in this repository.






This project was made for educational purposes and understanding how to apply transformers and diffusion models to audio.



# Running Code
## Generate Dataset

The ```process_audio_to_pt.py``` file can be used to translate your audio dataset into the pt files necessary for training the models in the repository.

For both the latest and previous versions of models you can change the configuration settings in the file directly. Run the models on "train" first to obtain the models, once they are finished you can then run on the "generate" mode.


## Note 
The parameters used in the training process were optimized for running on an HPC with an A100 with 40GB of VRAM, as well as higher memory usage for converting audio through encodec roughly 120GB of memory were used for the concurrency in the latest model to speed up encoding. 