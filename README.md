# CS6910-Assignment-3

This README file guides you through the code present in this repository and provides instructions on how to implement it correctly. For more details, refer to my WandB project: [CS6910-Assignment-3 Report](https://wandb.ai/sumanta_roy/CS6910_assignment_3/reports/CS6910-Assignment-3--Vmlldzo3OTI0ODcx).

## Root Directory

In the root of this GitHub repository, you will find two Python scripts, `train_noattention.py` and `train_attention.py`. Both scripts accept command-line arguments and perform the following tasks:

- **`train_noattention.py`**: This script trains a seq2seq model without an attention mechanism using the specified hyperparameters. If no arguments are provided, it uses default hyperparameters optimized through Bayesian hyperparameter tuning. The supported arguments are:

  | Name | Default Value | Description |
  | :---: | :-------------: | :----------- |
  | `-wp`, `--wandb_project` | `myprojectname` | Project name for tracking experiments in Weights & Biases dashboard |
  | `-we`, `--wandb_entity` | `myname` | Wandb Entity for tracking experiments in the Weights & Biases dashboard |
  | `-e`, `--epochs` | `10` | Number of training epochs |
  | `-b`, `--batch_size` | `64` | Batch size for training |
  | `-do`, `--dropout` | `0.3` | Dropout probability |
  | `-cell_type`, `--cell_type` | `LSTM` | Type of recurrent cell; choices: ["LSTM", "RNN", "GRU"] |
  | `-hs`, `--hidden_size` | `256` | Size of hidden units in the encoder-decoder networks |
  | `-nl`, `--n_layers` | `2` | Number of hidden encoder/decoder layers |
  | `-in_emb`, `--in_embed` | `256` | Length of input embedding |

- **`train_attention.py`**: This script trains a seq2seq model with an attention mechanism using the specified hyperparameters. If no arguments are provided, it uses default hyperparameters optimized through Bayesian hyperparameter tuning. The supported arguments are:

  | Name | Default Value | Description |
  | :---: | :-------------: | :----------- |
  | `-wp`, `--wandb_project` | `myprojectname` | Project name for tracking experiments in Weights & Biases dashboard |
  | `-we`, `--wandb_entity` | `myname` | Wandb Entity for tracking experiments in the Weights & Biases dashboard |
  | `-e`, `--epochs` | `10` | Number of training epochs |
  | `-b`, `--batch_size` | `256` | Batch size for training |
  | `-do`, `--dropout` | `0.2` | Dropout probability |
  | `-cell_type`, `--cell_type` | `GRU` | Type of recurrent cell; choices: ["LSTM", "RNN", "GRU"] |
  | `-hs`, `--hidden_size` | `256` | Size of hidden units in the encoder-decoder networks |
  | `-nl`, `--n_layers` | `3` | Number of hidden encoder/decoder layers |
  | `-in_emb`, `--in_embed` | `64` | Length of input embedding |

Running these scripts trains the respective seq2seq models using the specified hyperparameters. During training, the scripts display the training loss, validation loss, training accuracy, and validation accuracy at each epoch. 

For reference, a Jupyter notebook, `command_line_example.ipynb`, is included in the root directory. This notebook demonstrates how to call these two Python scripts and shows sample output. The two commands are:

```bash
!python train_noattention.py -wp CS6910_assignment_3 -we sumanta_roy
!python train_attention.py -wp CS6910_assignment_3 -we sumanta_roy
