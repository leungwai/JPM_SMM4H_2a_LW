import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from training_code import *
from load_data import initialize_data
from reading_datasets import read_task
from labels_to_ids import task7_labels_to_ids
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(model_load_location):
    max_len = 256
    batch_size = 32
    grad_step = 1
    learning_rate = 1e-05
    initialization_input = (max_len, batch_size)

    #Reading datasets and initializing data loaders
    dataset_location = '../Datasets/'
    test_data = read_task(dataset_location , split = 'dev')

    labels_to_ids = task7_labels_to_ids
    input_data = (test_data, labels_to_ids)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time

    tokenizer = AutoTokenizer.from_pretrained(model_load_location)
    model = AutoModelForSequenceClassification.from_pretrained(model_load_location)

    # unshuffled testing data
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    # Getting testing dataloaders
    test_loader = initialize_data(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = False)

    test_net_f1 = 0

    test_ind_f1 = [0, 0, 0]
    test_ind_precision = [0, 0, 0]
    test_ind_recall = [0, 0, 0]

    start = time.time()

    # Run the model with unshuffled testing data
    test_result, test_accuracy, test_fm_f1, test_fm_precision, test_fm_recall, test_saho_f1, test_saho_precision, test_saho_recall, test_sc_f1, test_sc_precision, test_sc_recall = testing(model, test_loader, labels_to_ids, device)
    print('DEV ACC:', test_accuracy)

    print(' ')
    print('fm test_f1:', test_fm_f1)
    print('fm test_precision:', test_fm_precision)
    print('fm test_recall:', test_fm_recall)

    print(' ')
    print('saho test_f1:', test_saho_f1)
    print('saho test_precision:', test_saho_precision)
    print('saho test_recall:', test_saho_recall)

    print(' ')
    print('sc test_f1:', test_sc_f1)
    print('sc test_precision:', test_sc_precision)
    print('sc test_recall:', test_sc_recall)

    test_ind_f1 = [test_fm_f1, test_saho_f1, test_sc_f1]
    test_ind_precision = [test_fm_precision, test_saho_precision, test_sc_precision]
    test_ind_recall = [test_fm_recall, test_saho_recall, test_sc_recall]

    # calculating the net f1 performance
    test_net_f1 = (1.0/3.0) * (test_fm_f1 + test_saho_f1 + test_sc_f1)

    print('TEST NET F1:', test_net_f1)

    now = time.time()

    print('TEST ACCURACY --> ', 'DEV:', round(test_accuracy, 5))
    print('TEST NET F1 --> ', 'DEV:', round(test_net_f1, 5))
    print('TIME PER EPOCH:', (now-start)/60 )
    print()

    return test_result, test_accuracy, test_net_f1, test_net_f1, test_ind_f1, test_ind_precision, test_ind_recall

if __name__ == '__main__':
    n_epochs = 10
    models = ['bert-base-uncased', 'roberta-base']

    # setting up the arrays to save data for all loops, models,

    # dev and test acc
    all_test_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    # factors to calculate final f1 performance metric
    all_ind_f1_score = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_ind_precision = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_ind_recall = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    # final f1 performance metric
    all_overall_f1_score = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    for loop_index in range(2):
        for model_name in models:
            test_print_statement = 'Testing ' + model_name + ' from loop ' + str(loop_index)
            print(test_print_statement)

            model_load_location = '../saved_models_2a/' + model_name + '/' + str(loop_index) + '/' 
            
            result_save_location = '../saved_test_result_2a/' + model_name + '/' + str(loop_index) + '/'
            
            unformatted_result_save_location = result_save_location + 'unformatted_test_result.tsv'
            formatted_result_save_location = result_save_location + 'formatted_test_result.tsv'

            test_result, test_acc, tb_acc, overall_f1_score, ind_f1_score, ind_precision, ind_recall = main(model_load_location)

            # Getting accuracy
            all_test_acc.at[loop_index, model_name] = test_acc

            # Getting best overall f1 score
            all_overall_f1_score.at[loop_index, model_name] = overall_f1_score
            
            # Getting best individual data (by category)
            all_ind_f1_score.at[loop_index, model_name] = ind_f1_score
            all_ind_precision.at[loop_index, model_name] = ind_precision
            all_ind_recall.at[loop_index, model_name] = ind_recall

            print("\n Testing results")
            print(test_result)
            formatted_test_result = test_result.drop(columns=['Orig'])

            os.makedirs(result_save_location, exist_ok=True)
            test_result.to_csv(unformatted_result_save_location, sep='\t', index=False)
            formatted_test_result.to_csv(formatted_result_save_location, sep='\t', index=False)

            print("Result files saved")

    print("\n All best overall f1 score")
    print(all_overall_f1_score)

    print("\n All best dev acc")
    print(all_test_acc)

    print("\n All best f1 score")
    print(all_ind_f1_score)

    print("\n All best precision")
    print(all_ind_precision)

    print("\n All best recall")
    print(all_ind_recall)   


    #saving all results into tsv

    os.makedirs('../testing_statistics/', exist_ok=True)
    all_overall_f1_score.to_csv('../testing_statistics/all_overall_f1_score.tsv', sep='\t')
    all_test_acc.to_csv('../testing_statistics/all_test_acc.tsv', sep='\t')
    all_ind_f1_score.to_csv('../testing_statistics/all_ind_f1_score.tsv', sep='\t')
    all_ind_precision.to_csv('../testing_statistics/all_ind_precision.tsv', sep='\t')
    all_ind_recall.to_csv('../testing_statistics/all_ind_recall.tsv', sep='\t')     

    print("Everything successfully completed")













    
        