#  OLMo 190M – custom project

This repo trains the [OLMo 190M](https://allenai.org/olmo) language model **locally** on a small chunk of Wikipedia.
Currently the repo is not using nor wandb or the full datasets since it is a test to debug the training pipeline.



## Project structure
 
  - `train.py`: Main entry point. Calls the run() function to kick off the entire pipeline — data loading, model building, training, and inference sampling.
  - `utils/dataloader.py`: Downloads and tokenizes a small chunk of Wikipedia using HuggingFace Datasets + Tokenizers. Prepares wiki_tokens.npy, builds a NumpyDataset, and returns a DataLoader.
  - `utils/model.py`: Builds the 190M parameter OLMo model using the TransformerConfig. Applies token ID 0 masking, sets up optimizer and training module (TransformerTrainModule).
  - `utils/inference.py`: Defines an InferenceCallback that runs during training. It generates text from the current model every N steps using a custom prompt (e.g. "The universe is ...").

## Project setup

1. Clone the olmo-core and place the olmo-core folder in the root of the project
https://github.com/allenai/OLMo-core/tree/main
2. Once cloned olmo core find folder OLMo-core/src/olmo_core and copy the folder to your root project folder

3. Set virtual env

```sh
chmod +x setup.sh
./setup.sh
```
## Run project


Run train.py

```sh
python3 train.py --steps 50 --batch-size 2 --prompt "The universe is "    
```

