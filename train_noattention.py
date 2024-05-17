#Importing the libraries:
import zipfile
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import gc
import random
import math
import wandb
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a character-level seq2seq model without attention')
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', help='Project name for Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='myname', help='Wandb Entity for Weights & Biases dashboard')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train the seq2seq model')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-do', '--dropout', type=int, default=0.3, help='Neuron dropout')
    parser.add_argument('-cell_type', '--cell_type', choices=['LSTM', 'GRU', 'RNN'], default='LSTM', help='Type of recurrent cell')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256, help='Size of hidden unit')
    parser.add_argument('-nl', '--n_layers', type=int, default=2, help='No of hidden encoder/decoder layers')
    parser.add_argument('-in_emb', '--in_embed', type=int, default=256, help='Length of input embedding')    
    args = parser.parse_args()
    return args

def main(args):

    # Initializing and runing WandB
    wandb.login(key='4734e60951ce310dbe17484eeeb5b3366b54850f')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    wandb.config.update(args)

    #Load data:
    zip_file_path = '/content/aksharantar_sampled.zip'
    extracted_folder_path = '/content/'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)
    extracted_folder_contents = os.listdir(extracted_folder_path)
    print("Contents of extracted folder:", extracted_folder_contents)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Trained on: " + str(device))
    train_dataset = pd.read_csv('/content/aksharantar_sampled/hin/hin_train.csv', names=['English', 'Hindi'], header=None)
    test_dataset = pd.read_csv('/content/aksharantar_sampled/hin/hin_test.csv', names=['English', 'Hindi'], header=None)
    val_dataset = pd.read_csv('/content/aksharantar_sampled/hin/hin_valid.csv', names=['English', 'Hindi'], header=None)

    #Support functions:
    def clear_gpu_cache():
        print("Initial GPU Usage")
        gpu_usage()
        torch.cuda.empty_cache()
        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)
        print("GPU Usage after emptying the cache")
        gpu_usage()
    def split_into_tokens(word):
        tokens = []
        for x in word:
            tokens.append(x)
        return tokens
    def encode_english(word):
        tokens = []
        for x in word:
            tokens.append(eng_dict[x])
        for x in range(len(tokens), max_english_length):
            tokens.append(eng_dict['<pad>'])
        return tokens
    def encode_hindi(word):
        tokens = []
        for x in word:
            tokens.append(hin_dict[x])
        tokens.append(hin_dict['<eow>'])
        for x in range(len(tokens), max_hindi_length + 1):
            tokens.append(hin_dict['<pad>'])
        return tokens
    def encode_test_english(word):
        tokens = []
        for x in word:
            tokens.append(eng_dict[x])
        for x in range(len(tokens), test_max_english_length):
            tokens.append(eng_dict['<pad>'])
        return tokens
    def encode_test_hindi(word):
        tokens = []
        for x in word:
            tokens.append(hin_dict[x])
        tokens.append(hin_dict['<eow>'])
        for x in range(len(tokens), test_max_hindi_length):
            tokens.append(hin_dict['<pad>'])
        return tokens
    def encode_val_english(word):
        tokens = []
        for x in word:
            tokens.append(eng_dict[x])
        for x in range(len(tokens), val_max_english_length):
            tokens.append(eng_dict['<pad>'])
        return tokens
    def encode_val_hindi(word):
        tokens = []
        for x in word:
            tokens.append(hin_dict[x])
        tokens.append(hin_dict['<eow>'])
        for x in range(len(tokens), val_max_hindi_length):
            tokens.append(hin_dict['<pad>'])
        return tokens
    def get_word(characters):
        return "".join(characters)
    def calculate_accuracy(target, predictions, flag):
        total = 0
        for x in range(len(target)):
            if torch.equal(target[x], predictions[x]):
                total += 1
        return total
    def translate_predictions(target, predictions, df):
        i = len(df)
        for x in range(len(predictions)):
            original = []
            for y in target[x]:
                if y != 1:
                    original.append(y)
                else:
                    break
            predicted = []
            for y in predictions[x]:
                if y != 1:
                    predicted.append(y)
                else:
                    break
            df.loc[i, ['Original']] = get_word([reverse_hin[x.item()] for x in original])
            df.loc[i, ['Predicted']] = get_word([reverse_hin[x.item()] for x in predicted])
            i += 1
        return df
    

    #Creating initial variables
    split_into_tokens(train_dataset.iloc[0]['Hindi'])
    max_english_length = 0
    max_hindi_length = 0
    test_max_english_length = 0
    test_max_hindi_length = 0
    for x in range(len(test_dataset)):
        temp = 0
        for y in test_dataset.iloc[x]['English']:
            temp += 1
        test_max_english_length = max(test_max_english_length, temp)
    for x in range(len(test_dataset)):
        temp = 0
        for y in test_dataset.iloc[x]['Hindi']:
            temp += 1
        test_max_hindi_length = max(test_max_hindi_length, temp)
    val_max_english_length = 0
    val_max_hindi_length = 0
    for x in range(len(val_dataset)):
        temp = 0
        for y in val_dataset.iloc[x]['English']:
            temp += 1
        val_max_english_length = max(val_max_english_length, temp)
    for x in range(len(val_dataset)):
        temp = 0
        for y in val_dataset.iloc[x]['Hindi']:
            temp += 1
        val_max_hindi_length = max(val_max_hindi_length, temp)
    english_vocab = []
    for x in range(len(train_dataset)):
        temp = 0
        for y in train_dataset.iloc[x]['English']:
            temp += 1
            if y not in english_vocab:
                english_vocab.append(y)
        if temp > max_english_length:
            max_english_length = max(max_english_length, temp)
    hindi_vocab = []
    for x in range(len(train_dataset)):
        temp = 0
        for y in train_dataset.iloc[x]['Hindi']:
            temp += 1
            if y not in hindi_vocab:
                hindi_vocab.append(y)
        max_hindi_length = max(temp, max_hindi_length)
    for x in range(len(test_dataset)):
        for y in test_dataset.iloc[x]['Hindi']:
            if y not in hindi_vocab:
                hindi_vocab.append(y)
    english_vocab = sorted(english_vocab)
    hindi_vocab = sorted(hindi_vocab)
    eng_dict = {}
    reverse_eng = {}
    for x in range(len(english_vocab)):
        eng_dict[english_vocab[x]] = x + 3
        reverse_eng[x + 3] = english_vocab[x]
    eng_dict['<sow>'] = 0
    eng_dict['<eow>'] = 1
    eng_dict['<pad>'] = 2
    reverse_eng[0] = '<sow>'
    reverse_eng[1] = '<eow>'
    reverse_eng[2] = '<pad>'
    hin_dict = {}
    reverse_hin = {}
    for x in range(len(hindi_vocab)):
        hin_dict[hindi_vocab[x]] = x + 3
        reverse_hin[x + 3] = hindi_vocab[x]
    hin_dict['<sow>'] = 0
    hin_dict['<eow>'] = 1
    hin_dict['<pad>'] = 2
    reverse_hin[0] = '<sow>'
    reverse_hin[1] = '<eow>'
    reverse_hin[2] = '<pad>'
    encode_english(train_dataset.iloc[0]['English'])
    eng_words = []
    hin_words = []
    for x in range(len(train_dataset)):
        eng_words.append(encode_english(train_dataset.iloc[x]['English']))
        hin_words.append(encode_hindi(train_dataset.iloc[x]['Hindi']))
    eng_words = torch.tensor(eng_words)
    hin_words = torch.tensor(hin_words)
    max_hindi_length
    max_hindi_length += 1
    test_max_hindi_length += 1
    val_max_hindi_length += 1
    max_hindi_length
    val_eng_words = []
    val_hin_words = []
    for x in range(len(val_dataset)):
        val_eng_words.append(encode_val_english(val_dataset.iloc[x]['English']))
        val_hin_words.append(encode_val_hindi(val_dataset.iloc[x]['Hindi']))
    val_eng_words = torch.tensor(val_eng_words)
    val_hin_words = torch.tensor(val_hin_words)
    test_eng_words = []
    test_hin_words = []
    for x in range(len(test_dataset)):
        test_eng_words.append(encode_test_english(test_dataset.iloc[x]['English']))
        test_hin_words.append(encode_test_hindi(test_dataset.iloc[x]['Hindi']))
    test_eng_words = torch.tensor(test_eng_words)
    test_hin_words = torch.tensor(test_hin_words)

    #No-attention-encoder-decoder:
    class Encoder(nn.Module):
        def __init__(self, char_embed_size, hidden_size, no_of_layers, dropout, rnn):
            super(Encoder, self).__init__()
            self.layer = no_of_layers
            self.rnn = rnn
            self.embedding = nn.Embedding(len(eng_dict), char_embed_size).to(device)
            self.embedding.weight.requires_grad = True
            self.drop = nn.Dropout(dropout)
            self.LSTM = nn.LSTM(char_embed_size, hidden_size, self.layer, batch_first=True, bidirectional=True).to(device)
            self.RNN = nn.RNN(char_embed_size, hidden_size, self.layer, batch_first=True, bidirectional=True).to(device)
            self.GRU = nn.GRU(char_embed_size, hidden_size, self.layer, batch_first=True, bidirectional=True).to(device)

        def forward(self, input, hidden, cell):
            embedded = self.embedding(input)
            embedded1 = self.drop(embedded)
            cell1 = cell
            if self.rnn == 'RNN':
                output, hidden1 = self.RNN(embedded1, hidden)
            elif self.rnn == 'LSTM':
                output, (hidden1, cell1) = self.LSTM(embedded1, (hidden, cell))
            elif self.rnn == 'GRU':
                output, hidden1 = self.GRU(embedded1, hidden)
            return output, (hidden1, cell1)


    class DecoderNoAttention(nn.Module):
        def __init__(self, char_embed_size, hidden_size, no_of_layers, dropout, batchsize, rnn):
            super(DecoderNoAttention, self).__init__()
            self.layer = no_of_layers
            self.batchsize = batchsize
            self.hidden_size = hidden_size
            self.rnn = rnn
            self.embedding = nn.Embedding(len(hin_dict), char_embed_size).to(device)
            self.drop = nn.Dropout(dropout)
            self.embedding.weight.requires_grad = True
            self.LSTM = nn.LSTM(char_embed_size + hidden_size * 2, hidden_size, self.layer, batch_first=True).to(device)
            self.RNN = nn.RNN(char_embed_size + hidden_size * 2, hidden_size, self.layer, batch_first=True).to(device)
            self.GRU = nn.GRU(char_embed_size + hidden_size * 2, hidden_size, self.layer, batch_first=True).to(device)
            self.linear = nn.Linear(hidden_size, len(hin_dict), bias=True).to(device)
            self.softmax = nn.Softmax(dim=2).to(device)

        def forward(self, input, hidden, cell, og_hidden, matrix):
            embedded = self.embedding(input)
            s1 = og_hidden.size()[1]
            s2 = og_hidden.size()[2]
            embedded1 = torch.cat((embedded, og_hidden[0].resize(s1, 1, s2), og_hidden[1].resize(s1, 1, s2)), dim=2)
            embedded2 = self.drop(embedded1)
            cell1 = cell
            if self.rnn == 'LSTM':
                output, (hidden1, cell1) = self.LSTM(embedded2, (hidden, cell))
            elif self.rnn == 'RNN':
                output, hidden1 = self.RNN(embedded2, hidden)
            elif self.rnn == 'GRU':
                output, hidden1 = self.GRU(embedded2, hidden)
            output1 = self.linear(output)
            return output1, (hidden1, cell1)

    #Evaluation and training functions:
    def val_evaluate(attention, val_eng_words, val_hin_words, encoder, decoder, batch_size, hidden_size, char_embed_size, no_of_layers):
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for x in range(0, len(val_dataset), batch_size):
                loss = 0
                input_tensor = val_eng_words[x:x + batch_size].to(device)
                if input_tensor.size()[0] < batch_size:
                    break
                en_hidden = torch.zeros(2 * no_of_layers, batch_size, hidden_size).to(device)
                en_cell = torch.zeros(2 * no_of_layers, batch_size, hidden_size).to(device)
                output, (hidden, cell) = encoder.forward(input_tensor, en_hidden, en_cell)
                del input_tensor
                del en_hidden
                del en_cell
                output = torch.split(output, [hidden_size, hidden_size], dim=2)
                output = torch.add(output[0], output[1]) / 2
                input2 = []
                for y in range(batch_size):
                    input2.append([0])
                input2 = torch.tensor(input2).to(device)
                hidden = hidden.resize(2, no_of_layers, batch_size, hidden_size)
                hidden1 = torch.add(hidden[0], hidden[1]) / 2
                cell = cell.resize(2, no_of_layers, batch_size, hidden_size)
                cell1 = torch.add(cell[0], cell[1]) / 2
                OGhidden = hidden1
                predicted = []
                predictions = []
                if attention:
                    temp = output
                else:
                    temp = OGhidden
                for i in range(val_max_hindi_length):
                    output1, (hidden1, cell1) = decoder.forward(input2, hidden1, cell1, temp, False)
                    predicted.append(output1)
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2, dim=2)
                    predictions.append(output3)
                    input2 = output3
                predicted = torch.cat(tuple(x for x in predicted), dim=1).to(device).resize(val_max_hindi_length * batch_size, len(hin_dict))
                predictions = torch.cat(tuple(x for x in predictions), dim=1).to(device)
                total_acc += calculate_accuracy(val_hin_words[x:x + batch_size].to(device), predictions, x)
                loss = nn.CrossEntropyLoss(reduction='sum')(predicted, val_hin_words[x:x + batch_size].reshape(-1).to(device))
                with torch.no_grad():
                    total_loss += loss.item()
            validation_loss = total_loss / (len(val_dataset) * val_max_hindi_length)
            validation_accuracy = (total_acc / len(val_dataset)) * 100
            del predictions
            del predicted
            del input2
            del output1
            del output2
            del output3
            del hidden1
            del cell1
            del OGhidden
            del output
            del cell
            return validation_loss, validation_accuracy


    def train(batch_size, hidden_size, char_embed_size, no_of_layers, dropout, epochs, rnn):
        gc.collect()
        torch.autograd.set_detect_anomaly(True)
        encoder = Encoder(char_embed_size, hidden_size, no_of_layers, dropout, rnn).to(device)
        decoder = DecoderNoAttention(char_embed_size, hidden_size, no_of_layers, dropout, batch_size, rnn).to(device)
        # print(encoder.parameters)
        # print(decoder.parameters)
        opt_encoder = optim.Adam(encoder.parameters(), lr=0.001)
        opt_decoder = optim.Adam(decoder.parameters(), lr=0.001)
        teacher_ratio = 0.5
        epoch_count = 0
        for _ in range(epochs):
            total_loss = 0
            total_acc = 0
            for x in range(0, len(train_dataset), batch_size):
                loss = 0
                opt_encoder.zero_grad()
                opt_decoder.zero_grad()
                input_tensor = eng_words[x:x + batch_size].to(device)
                # taking initial hidden and cell states as (2* no_of_layers, hidden_size, hidden_size) because I have considered encoder to be bidirectional
                en_hidden = torch.zeros(2 * no_of_layers, batch_size, hidden_size).to(device)
                en_cell = torch.zeros(2 * no_of_layers, batch_size, hidden_size).to(device)
                if input_tensor.size()[0] < batch_size:
                    break
                output, (hidden, cell) = encoder.forward(input_tensor, en_hidden, en_cell)
                del en_hidden
                del en_cell
                del input_tensor
                input2 = []
                for y in range(batch_size):
                    input2.append([0])
                input2 = torch.tensor(input2).to(device)
                hidden = hidden.resize(2, no_of_layers, batch_size, hidden_size)
                cell = cell.resize(2, no_of_layers, batch_size, hidden_size)
                # averaging due to bidirectional encoder
                hidden1 = torch.add(hidden[0], hidden[1]) / 2
                cell1 = torch.add(cell[0], cell[1]) / 2
                OGhidden = hidden1
                predicted = []
                predictions = []
                use_teacher_forcing = True if random.random() < teacher_ratio else False
                if use_teacher_forcing:
                    for i in range(max_hindi_length):
                        output1, (hidden1, cell1) = decoder.forward(input2, hidden1, cell1, OGhidden, False)
                        predicted.append(output1)
                        output2 = decoder.softmax(output1)
                        output3 = torch.argmax(output2, dim=2)
                        predictions.append(output3)
                        input2 = hin_words[x:x + batch_size, i].to(device).resize(batch_size, 1)
                else:
                    for i in range(max_hindi_length):
                        output1, (hidden1, cell1) = decoder.forward(input2, hidden1, cell1, OGhidden, False)
                        predicted.append(output1)
                        output2 = decoder.softmax(output1)
                        output3 = torch.argmax(output2, dim=2)
                        predictions.append(output3)
                        input2 = output3
                predicted = torch.cat(tuple(x for x in predicted), dim=1).to(device).resize(max_hindi_length * batch_size, len(hin_dict))
                predictions = torch.cat(tuple(x for x in predictions), dim=1).to(device)
                total_acc += calculate_accuracy(hin_words[x:x + batch_size].to(device), predictions, x)
                loss = nn.CrossEntropyLoss(reduction='sum')(predicted, hin_words[x:x + batch_size].reshape(-1).to(device))
                with torch.no_grad():
                    total_loss += loss.item()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
                opt_encoder.step()
                opt_decoder.step()
            del predictions
            del predicted
            del input2
            del output1
            del output2
            del output3
            del hidden1
            del cell1
            del OGhidden
            del output
            del cell
            training_loss = total_loss / (51200 * max_hindi_length)
            training_accuracy = total_acc / 512
            validation_loss, validation_accuracy = val_evaluate(False, val_eng_words, val_hin_words, encoder, decoder, batch_size, hidden_size, char_embed_size, no_of_layers)
            wandb.log({'training_accuracy': training_accuracy, 'validation_accuracy': validation_accuracy, 'training_loss': training_loss, 'validation_loss': validation_loss, 'epoch': epoch_count + 1})
            print("Epoch: " + str(epoch_count + 1) + "/" + str(epochs) + "; Train loss: " + str(training_loss) + "; Val loss: " + str(validation_loss))
            print("Training accuracy: " + str(training_accuracy) + "; Validation accuracy: " + str(validation_accuracy))
            epoch_count += 1
        return encoder, decoder           


    #Taking args inputs:
    hidden_size = args.hidden_size
    char_embed_size = args.in_embed
    no_of_layers = args.n_layers
    dropout = args.dropout
    rnn = args.cell_type
    epochs = args.epochs
    batchsize = args.batch_size
    Encoder1,Decoder1 = train(batchsize,hidden_size,char_embed_size,no_of_layers,dropout,epochs,rnn)
 
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
