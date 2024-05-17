# CS6910-Assignment-3

This README file guides you through all the code present in this repository and provides instructions on how to implement it correctly.
Also, link to my WandB project: [https://wandb.ai/sumanta_roy/CS6910_assignment_1/reports/Sumanta-s-CS6910-Assignment-1--Vmlldzo3MTU4NTE5](https://wandb.ai/sumanta_roy/CS6910_assignment_3/reports/CS6910-Assignment-3--Vmlldzo3OTI0ODcx)

I will guide you though all the code that is present in this repository and how to use each one of them.

## Root Directory

In the root GitHub repository, you will find two python scripts, named `train_noattention.py` and `train_attention.py`. Both of these scripts accept command line arguments, and does the followig tasks:

- `train_noattention.py`: It takes command-line arguments the hyper-parameters of a seq-2-seq model with no attention mechanism. If you don't explicitly provide any command line arguments, it will select the default hyper-parameters (which are the best hyper-parameters, leading to the highest accuracy found using a Bayesian hyperparameter tuning search). The arguments that it supports are:

  | Name | Default Value | Description |
  | :---: | :-------------: | :----------- |
  | `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
  | `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
  | `-e`, `--epochs` | 10 | Choose the number of epochs for training |
  | `-b`, `--batch_size` | 64 | Choose the batch size for training the model |
  | `-do`, `--dropout` | 0.3 | Choose the neuron dropout probability |
  | `-cell_type`, `--cell_type` | 'LSTM' | Choose the type of recurrent cell; choices: ["LSTM", "RNN", "GRU"]|
  | `-hs`, `--hidden_size` | 256 | Choose the size of hidden units in the encoder-decoder networks|
  | `-nl`, `--n_layers` | 2 | Choose the number of hidden encoder/decoder layers|
  | `-in_emb`, `--in_embed` | 256 | Choose the length of input embedding| 


