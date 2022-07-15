import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from training_code import *
from load_data import initialize_eval_test
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
    test_loader= initialize_eval_test(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = False)

    start = time.time()

    # Run the model with unshuffled testing data
    overall_prediction_data, eval_accuracy, fm_f1_score, fm_precision, fm_recall, saho_f1_score, saho_precision, saho_recall, sc_f1_score, sc_precision, sc_recall, overall_fm_cr_df, overall_fm_cm_df, overall_saho_cr_df, overall_saho_cm_df, overall_sc_cr_df, overall_sc_cm_df, eval_logits = val_testing(model, test_loader, labels_to_ids, device)

    net_f1 = (1.0/3.0) * (fm_f1_score + saho_f1_score + sc_f1_score)

    print('EVAL TEST ACC:', eval_accuracy)
    print('EVAL TEST F1:', net_f1)

    now = time.time()

    print('TIME TO COMPLETE:', (now-start)/60 )
    print()

    return overall_prediction_data, eval_accuracy, net_f1, fm_f1_score, fm_precision, fm_recall, saho_f1_score, saho_precision, saho_recall, sc_f1_score, sc_precision, sc_recall, overall_fm_cr_df, overall_fm_cm_df, overall_saho_cr_df, overall_saho_cm_df, overall_sc_cr_df, overall_sc_cm_df, eval_logits



if __name__ == '__main__':
    train_val_start_time = time.time()
    n_rounds = 5
    models = ['bert-large-uncased', 'roberta-large']

    # setting up the arrays to save data for all loops, models, and epochs
    # accuracy
    all_best_acc = pd.DataFrame(index=range(n_rounds), columns=models)
    all_best_f1_score = pd.DataFrame(index=range(n_rounds), columns=models)

    all_best_ind_f1_score = pd.DataFrame(index=range(n_rounds), columns=models)
    all_best_ind_precision = pd.DataFrame(index=range(n_rounds), columns=models)
    all_best_ind_recall = pd.DataFrame(index=range(n_rounds), columns=models)

    for loop_index in range(n_rounds):
        for model_name in models:
            test_print_statement = 'Testing ' + model_name + ' from loop ' + str(loop_index)
            print(test_print_statement)

            model_load_location = '../20_epochs_large_model/saved_models_2a/' + model_name + '/' + str(loop_index) + '/' 
            
            result_save_location = '../20_epochs_large_model/eval_testing/saved_eval_test_result_2a/' + model_name + '/' + str(loop_index) + '/'
            report_result_save_location = '../20_epochs_large_model/eval_testing/saved_eval_report_2a/' + model_name + '/' + str(loop_index) + '/'

            unformatted_result_save_location = result_save_location + 'unformatted_eval_test_result.tsv'
            formatted_result_save_location = result_save_location + 'formatted_eval_test_result.tsv'

            overall_prediction_data, eval_accuracy, net_f1, fm_f1_score, fm_precision, fm_recall, saho_f1_score, saho_precision, saho_recall, sc_f1_score, sc_precision, sc_recall, overall_fm_cr_df, overall_fm_cm_df, overall_saho_cr_df, overall_saho_cm_df, overall_sc_cr_df, overall_sc_cm_df, eval_logits = main(model_load_location)

            # Getting best f1, precision, and recall, accuracy
            all_best_acc.at[loop_index, model_name] = eval_accuracy
            all_best_f1_score.at[loop_index, model_name] = net_f1
            
            all_best_ind_f1_score.at[loop_index, model_name] = [fm_f1_score, saho_f1_score, sc_f1_score]
            all_best_ind_precision.at[loop_index, model_name] = [fm_precision, saho_precision, sc_precision]
            all_best_ind_recall.at[loop_index, model_name] = [fm_recall, saho_recall, sc_recall]

            os.makedirs(report_result_save_location, exist_ok=True)
            cr_df_location = report_result_save_location + 'classification_report.tsv'
            cm_df_location = report_result_save_location + 'confusion_matrix.tsv'
            eval_logits_location = report_result_save_location + 'eval_logits.tsv'
            
            format_eval_logits = pd.DataFrame(eval_logits, columns=['0', '1', '2'])
            format_eval_logits.to_csv(eval_logits_location, sep='\t')
        

            os.makedirs(report_result_save_location, exist_ok=True)
            fm_cr_df_location = report_result_save_location + 'fm_eval_test_classification_report.tsv'
            fm_cm_df_location = report_result_save_location + 'fm_eval_test_confusion_matrix.tsv'
        
            saho_cr_df_location = report_result_save_location + 'saho_eval_test_classification_report.tsv'
            saho_cm_df_location = report_result_save_location + 'saho_eval_test_confusion_matrix.tsv'
        
            sc_cr_df_location = report_result_save_location + 'sc_eval_test_classification_report.tsv'
            sc_cm_df_location = report_result_save_location + 'sc_eval_test_confusion_matrix.tsv'

            overall_fm_cr_df.to_csv(fm_cr_df_location, sep='\t')
            overall_fm_cm_df.to_csv(fm_cm_df_location, sep='\t')
            overall_saho_cr_df.to_csv(saho_cr_df_location, sep='\t')
            overall_saho_cm_df.to_csv(saho_cm_df_location, sep='\t')
            overall_sc_cr_df.to_csv(sc_cr_df_location, sep='\t')
            overall_sc_cm_df.to_csv(sc_cm_df_location, sep='\t')

            print("\n Testing results")
            print(overall_prediction_data)
            formatted_test_result = overall_prediction_data.drop(columns=['text'])

            os.makedirs(result_save_location, exist_ok=True)
            overall_prediction_data.to_csv(unformatted_result_save_location, sep='\t', index=False)
            formatted_test_result.to_csv(formatted_result_save_location, sep='\t', index=False)

            print("Result files saved")

     # printing results for analysis
    print("\n All best overall f1 score")
    print(all_best_f1_score)

    print("\n All best dev acc")
    print(all_best_acc)

    print("\n All best f1 score")
    print(all_best_ind_f1_score)

    print("\n All best precision")
    print(all_best_ind_precision)

    print("\n All best recall")
    print(all_best_ind_recall)

    #saving all results into tsv

    os.makedirs('../20_epochs_large_model/eval_testing/validation_stats/', exist_ok=True)
    all_best_f1_score.to_csv('../20_epochs_large_model/eval_testing/validation_stats/all_best_overall_f1_score.tsv', sep='\t')
    all_best_acc.to_csv('../20_epochs_large_model/eval_testing/validation_stats/all_best_dev_acc.tsv', sep='\t')
    all_best_ind_f1_score.to_csv('../20_epochs_large_model/eval_testing/validation_stats/all_best_ind_f1_score.tsv', sep='\t')
    all_best_ind_precision.to_csv('../20_epochs_large_model/eval_testing/validation_stats/all_best_ind_precision.tsv', sep='\t')
    all_best_ind_recall.to_csv('../20_epochs_large_model/eval_testing/validation_stats/all_best_ind_recall.tsv', sep='\t')

    train_val_end_time = time.time()
    
    total_time = (train_val_end_time - train_val_start_time) / 60
    print("Everything successfully completed")
    print("Time to complete:", total_time)

    with open('../20_epochs_large_model/eval_testing/validation_stats/time.txt', 'w') as file:
        file.write("Time to complete: ")
        file.write(str(total_time))
        file.write(" mins")











    
        