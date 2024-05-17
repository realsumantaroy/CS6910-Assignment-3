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

This folder contains the solution to question 1. There is a jupyter notebook named `Question_1.ipynb`, which gives a simple structure of an encoder-decoder based seq2seq model with no attention mechanicsm. The code is very modular and user-friendly, where the user can choose the various hyper-parameters like, input embedding size, dimension of hidden layer (states), the dropout probability, etc. 

## /Question 2

This folder contains the solution to question 2. There is a jupyter notebook named `Question_2.ipynb`, which used the encoder-decoder based seq2seq model with no attention mechanicsm as defined in Question 1, and then also defines a train function which is used to train the optimizer. A Bayesian hyperparameter tuning is conducted using the WandB's sweep functionality. The notebook contains the the system generated logs and the hyper-parameter tuning in details. If you want to see how each and every run in each sweep performed, you can easily navigate to that particular step and click on the WandB link to see that log in detail. My WandB project is public for you to view.

## /Question 3

This folder contains a Jupyter notebook named `Question_3.ipynb`, which uses the best hyper-parameter from the Sweep conducted last time and then uses those hyperparameters to train the model again and then the accuracy of the model is checked on the test data.

## /Question 5

This folder contains a Jupyter notebook named `Question 5 sweep attention.ipynb`, which introces an attention mechanism to the vanilla encoder-decoder based seq2seq model (as in Questions 1, 2 and 3). Then a Bayesian hyper-parameter tuning is conducted using the sweep functionality of WandB, just like last time, for this attention based model. 

The best hyper-parameters are chosen from the sweep. In the same folder you will also find a jupyter notebook named `Question 5 test accuracy + att heat map.ipynb`, which uses the best hyperparameters found to re-train the model again, and the accuracy of the model is checked on the test data. Moreover, 10 sample words are taken from the test set and the attention matrix (heat-map) is plotted for each. heatmap can be found in the jupyter notebook itself, as well as in Question 5/attention_matrix.png. 

In addition to that, an Excell file is created titled `predictions_attention.xlsx`, which compares the predicted word and the original work for all of the test cases. This excell file can be found in predictions_attention/predictions_attention.xlsx.

It is to be noted that all of the three tasks: computing the test accuracy, plotting the heat map for attention and the excell file generation, the code can be found in the notebook `Question 5 test accuracy + att heat map.ipynb` itself.




This folder contains a Jupyter notebook named `Question_5_sweep_attention.ipynb`, which introduces an attention mechanism to the vanilla encoder-decoder-based seq2seq model (as in Questions 1, 2, and 3). Bayesian hyperparameter tuning is conducted using WandB's sweep functionality for this attention-based model. The best hyperparameters are chosen from the sweep. 

In the same folder, you will also find a Jupyter notebook named `Question_5_test_accuracy_and_att_heatmap.ipynb`, which re-trains the model using the best hyperparameters and evaluates its accuracy on the test data. Additionally, 10 sample words from the test set are used to plot the attention matrix (heatmap) for each. The heatmaps can be found in the Jupyter Notebook itself, as well as in [Link](Question_5/attention_matrix.png). Furthermore, an Excel file titled predictions_attention.xlsx is created, comparing the predicted words and the original words for all the test cases. This Excel file can be found in [Link](predictions_attention/predictions_attention.xlsx).

All tasks, including computing test accuracy, plotting the attention heatmap, and generating the Excel file, are covered in the notebook Question_5_test_accuracy_and_att_heatmap.ipynb.

## /predictions_attention

Contains the Excel file `predictions_attention.xlsx` as described in the previous section.

## Note
If you have any further questions in any part of the assignment, please feel free to contact me at ce22s003@smail.iitm.ac.in (+919083782161).

