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

Running these scripts trains the respective seq2seq models using the specified hyperparameters (through the command line arguments). During training, the scripts display the training loss, validation loss, training accuracy, and validation accuracy at each epoch. 

For reference, a Jupyter notebook, `command_line_example.ipynb`, is included in the root directory. This notebook demonstrates how to call these two Python scripts and shows sample output. For example, two sample commands are:

```bash
!python train_noattention.py -wp CS6910_assignment_3 -we sumanta_roy
!python train_attention.py -wp CS6910_assignment_3 -we sumanta_roy
```

## /Question 1

This folder contains the solution to question 1. There is a jupyter notebook named `Question_1.ipynb`, which provides a simple structure of an encoder-decoder-based seq2seq model with no attention mechanism. The code is modular and user-friendly, allowing users to choose various hyperparameters such as input embedding size, dimension of hidden layers (states), dropout probability, etc.

## /Question 2

This folder contains the solution to Question 2. There is a jupyter notebook named `Question_2.ipynb`, which uses the encoder-decoder-based seq2seq model without attention mechanism as defined in Question 1. Additionally, it defines a training function used to train the optimizer. Bayesian hyperparameter tuning is conducted using WandB's sweep functionality. The notebook includes system-generated logs and detailed hyperparameter tuning results. Each run's performance in the sweep can be viewed by navigating to the corresponding WandB link. My WandB project is public for you to view.

## /Question 3

This folder contains a Jupyter notebook named `Question_3.ipynb`, which uses the best hyper-parameter from the Sweep conducted previously. The model is re-trained using these optimal hyperparameters, and its accuracy is evaluated on the test data.


## /Question 5

This folder contains a Jupyter notebook named `Question_5_sweep_attention.ipynb`, which introduces an attention mechanism to the vanilla encoder-decoder-based seq2seq model (as in Questions 1, 2, and 3). Bayesian hyperparameter tuning is conducted using WandB's sweep functionality for this attention-based model. The best hyperparameters are chosen from the sweep.

In the same folder, you will also find a Jupyter notebook named `Question_5_test_accuracy_and_att_heatmap.ipynb`, which re-trains the model using the best hyperparameters and evaluates its accuracy on the test data. Additionally, 10 sample words from the test set are used to plot the attention matrix (heatmap) for each. The heatmaps can be found in the Jupyter notebook itself, as well as inside the folder. Furthermore, an Excel file titled `predictions_attention.xlsx` is created, comparing the predicted words and the original words for all the test cases. This Excel file can be found in [this link](predictions_attention/predictions_attention.xlsx).

All tasks, including computing test accuracy, plotting the attention heatmap, and generating the Excel file, are covered in the notebook `Question_5_test_accuracy_and_att_heatmap.ipynb`.


## /predictions_attention

Contains the Excel file `predictions_attention.xlsx` as described in the previous section.

## Note
If you have any further questions in any part of the assignment, please feel free to contact me at ce22s003@smail.iitm.ac.in (+919083782161).

