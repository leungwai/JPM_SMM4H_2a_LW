import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from load_data import initialize_data
from reading_datasets import read_task
from labels_to_ids import task7_labels_to_ids
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train(epoch, training_loader, model, optimizer, device, grad_step = 1, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    optimizer.zero_grad()
    
    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        if (idx + 1) % 20 == 0:
            print('FINSIHED BATCH:', idx, 'of', len(training_loader))

        #loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += output[0]

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )
        
        # backward pass
        output['loss'].backward()
        if (idx + 1) % grad_step == 0:
            optimizer.step()
            optimizer.zero_grad()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    #print(f"Training loss epoch: {epoch_loss}")
    #print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def testing(model, testing_loader, labels_to_ids, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    
    eval_fm_f1, eval_fm_precision, eval_fm_recall = 0, 0, 0
    eval_saho_f1, eval_saho_precision, eval_saho_recall = 0, 0, 0
    eval_sc_f1, eval_sc_precision, eval_sc_recall = 0, 0, 0

    eval_preds, eval_labels = [], []

    eval_tweet_ids, eval_topics, eval_orig_sentences = [], [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)

            # to attach back to prediction data later 
            tweet_ids = batch['tweet_id']
            topics = batch['topic']
            orig_sentences = batch['orig_sentence']
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            eval_loss += output['loss'].item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            eval_tweet_ids.extend(tweet_ids)
            eval_topics.extend(topics)
            eval_orig_sentences.extend(orig_sentences)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

            test_label = [id.item() for id in labels]
            test_pred = [id.item() for id in predictions]

            batch_prediction_data = pd.DataFrame(zip(tweet_ids, orig_sentences, topics, test_label, test_pred), columns=['id', 'text', 'Claim', 'Orig', 'Stance'])
            
            temp_fm_f1_score, temp_fm_precision, temp_fm_recall, temp_saho_f1_score, temp_saho_precision, temp_saho_recall, temp_sc_f1_score, temp_sc_precision, temp_sc_recall = calculate_f1(batch_prediction_data)
            
            eval_fm_f1 += temp_fm_f1_score
            eval_fm_precision += temp_fm_precision
            eval_fm_recall += temp_fm_recall
            
            eval_saho_f1 += temp_saho_f1_score
            eval_saho_precision += temp_saho_precision
            eval_saho_recall += temp_saho_recall
            
            eval_sc_f1 += temp_sc_f1_score
            eval_sc_precision += temp_sc_precision
            eval_sc_recall += temp_sc_recall


    # labels = [ids_to_labels[id.item()] for id in eval_labels]
    # predictions = [ids_to_labels[id.item()] for id in eval_preds]

    labels = [id.item() for id in eval_labels]
    predictions = [id.item() for id in eval_preds]
    
    
    # Calculating the f1 score, precision, and recall separately  by breaking the data apart 
    overall_prediction_data = pd.DataFrame(zip(eval_tweet_ids, eval_orig_sentences, eval_topics, labels, predictions), columns=['id', 'text', 'Claim', 'Orig', 'Stance'])

    overall_fm_classification_report, overall_fm_confusion_matrix, overall_saho_classification_report, overall_saho_confusion_matrix, overall_sc_classification_report, overall_sc_confusion_matrix = calculate_overall_f1(overall_prediction_data)
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    
    eval_fm_f1 = eval_fm_f1 / nb_eval_steps
    eval_fm_precision = eval_fm_precision / nb_eval_steps
    eval_fm_recall = eval_fm_recall / nb_eval_steps

    eval_saho_f1 = eval_saho_f1 / nb_eval_steps
    eval_saho_precision = eval_saho_precision / nb_eval_steps
    eval_saho_recall = eval_saho_recall / nb_eval_steps

    eval_sc_f1 = eval_sc_f1 / nb_eval_steps
    eval_sc_precision = eval_sc_precision / nb_eval_steps
    eval_sc_recall = eval_sc_recall / nb_eval_steps

    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return overall_prediction_data, eval_accuracy, eval_fm_f1, eval_fm_precision, eval_fm_recall, eval_saho_f1, eval_saho_precision, eval_saho_recall, eval_sc_f1, eval_sc_precision, eval_sc_recall, overall_fm_classification_report, overall_fm_confusion_matrix, overall_saho_classification_report, overall_saho_confusion_matrix, overall_sc_classification_report, overall_sc_confusion_matrix


def calculate_f1(prediction_data):
    fm_df = prediction_data.loc[prediction_data['Claim'] == 'face masks']
    saho_df = prediction_data.loc[prediction_data['Claim'] == 'stay at home orders']
    sc_df = prediction_data.loc[prediction_data['Claim'] == 'school closures']


    # splitting data into label and prediction of respective classes
    fm_label = fm_df['Orig'].tolist()
    fm_pred = fm_df['Stance'].tolist()

    saho_label = saho_df['Orig'].tolist()
    saho_pred = saho_df['Stance'].tolist()

    sc_label = sc_df['Orig'].tolist()
    sc_pred = sc_df['Stance'].tolist()

    # running performance metrics of each class
    print("Running performance metrics")
    fm_f1_score = f1_score(fm_label, fm_pred, labels=[0,1], average='macro')
    fm_precision = precision_score(fm_label, fm_pred, labels=[0,1], average='macro')
    fm_recall = recall_score(fm_label, fm_pred, labels=[0,1], average='macro')

    saho_f1_score = f1_score(saho_label, saho_pred, labels=[0,1], average='macro')
    saho_precision = precision_score(saho_label, saho_pred, labels=[0,1], average='macro')
    saho_recall = recall_score(saho_label, saho_pred, labels=[0,1], average='macro')

    sc_f1_score = f1_score(sc_label, sc_pred, labels=[0,1], average='macro')
    sc_precision = precision_score(sc_label, sc_pred, labels=[0,1], average='macro')
    sc_recall = recall_score(sc_label, sc_pred, labels=[0,1], average='macro')

    print("Finished running performance metrics")
    return fm_f1_score, fm_precision, fm_recall, saho_f1_score, saho_precision, saho_recall, sc_f1_score, sc_precision, sc_recall


def calculate_overall_f1(prediction_data):
    fm_df = prediction_data.loc[prediction_data['Claim'] == 'face masks']
    saho_df = prediction_data.loc[prediction_data['Claim'] == 'stay at home orders']
    sc_df = prediction_data.loc[prediction_data['Claim'] == 'school closures']

    # splitting data into label and prediction of respective classes
    fm_label = fm_df['Orig'].tolist()
    fm_pred = fm_df['Stance'].tolist()

    saho_label = saho_df['Orig'].tolist()
    saho_pred = saho_df['Stance'].tolist()

    sc_label = sc_df['Orig'].tolist()
    sc_pred = sc_df['Stance'].tolist()

    # running performance metrics of each class
    print("Running overall performance metrics")
    fm_classification_report = classification_report(fm_label, fm_pred)
    fm_confusion_matrix = confusion_matrix(fm_label, fm_pred)

    saho_classification_report = classification_report(saho_label, saho_pred)
    saho_confusion_matrix = confusion_matrix(saho_label, saho_pred)

    sc_classification_report = classification_report(sc_label, sc_pred)
    sc_confusion_matrix = confusion_matrix(sc_label, sc_pred)


    print("Finished running overall performance metrics")
    return fm_classification_report, fm_confusion_matrix, saho_classification_report, saho_confusion_matrix, sc_classification_report, sc_confusion_matrix

    

def main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location):
    #Initialization training parameters
    max_len = 256
    batch_size = 32
    grad_step = 1
    learning_rate = 1e-05
    initialization_input = (max_len, batch_size)

    #Reading datasets and initializing data loaders
    dataset_location = '../Datasets/'

    train_data = read_task(dataset_location , split = 'train')
    dev_data = read_task(dataset_location , split = 'dev')
    #test_data = read_task(dataset_location , split = 'dev')#load test set
    labels_to_ids = task7_labels_to_ids
    input_data = (train_data, dev_data, labels_to_ids)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    if model_load_flag:
        tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        model = AutoModelForSequenceClassification.from_pretrained(model_load_location)
    else: 
        tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader = initialize_data(tokenizer, initialization_input, train_data, labels_to_ids, shuffle = True)
    dev_loader = initialize_data(tokenizer, initialization_input, dev_data, labels_to_ids, shuffle = True)
    #test_loader = initialize_data(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = True)#create test loader

    best_overall_prediction_data = []
    best_dev_acc = 0
    best_test_acc = 0
    best_epoch = -1
    best_tb_acc = 0
    best_tb_epoch = -1

    best_net_f1 = 0
    best_ind_f1 = [0,0,0]
    best_ind_precision = [0,0,0]
    best_ind_recall = [0,0,0]

    all_epoch_data = pd.DataFrame(index=[0,1,2,3,4,5,6,7,8,9], columns=['overall_f1', 'dev_accuracy', 'fm_f1', 'fm_precision', 'fm_recall', 'saho_f1', 'saho_precision', 'saho_recall', 'sc_f1', 'sc_precision', 'sc_recall'])

    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train model
        model = train(epoch, train_loader, model, optimizer, device, grad_step)
        
        #testing and logging
        dev_overall_prediction, dev_accuracy, dev_fm_f1, dev_fm_precision, dev_fm_recall, dev_saho_f1, dev_saho_precision, dev_saho_recall, dev_sc_f1, dev_sc_precision, dev_sc_recall, overall_fm_classification_report, overall_fm_confusion_matrix, overall_saho_classification_report, overall_saho_confusion_matrix, overall_sc_classification_report, overall_sc_confusion_matrix = testing(model, dev_loader, labels_to_ids, device)

        print('DEV ACC:', dev_accuracy)
        
        print(' ')
        print('fm dev_f1:', dev_fm_f1)
        print('fm dev_precision:', dev_fm_precision)
        print('fm dev_recall:', dev_fm_recall)

        print(' ')
        print('saho dev_f1:', dev_saho_f1)
        print('saho dev_precision:', dev_saho_precision)
        print('saho dev_recall:', dev_saho_recall)

        print(' ')
        print('sc dev_f1:', dev_sc_f1)
        print('sc dev_precision:', dev_sc_precision)
        print('sc dev_recall:', dev_sc_recall)

        # calculating the net f1 performance
        dev_net_f1 = (1.0/3.0) * (dev_fm_f1 + dev_saho_f1 + dev_sc_f1)

        print('NET F1:', dev_net_f1)
        
        
        #labels_test, predictions_test, test_accuracy = testing(model, test_loader, labels_to_ids, device)
        #print('TEST ACC:', test_accuracy)

        all_epoch_data.at[epoch, 'overall_f1'] = dev_net_f1
        all_epoch_data.at[epoch, 'fm_accuracy'] = dev_accuracy

        all_epoch_data.at[epoch, 'fm_f1'] = dev_fm_f1
        all_epoch_data.at[epoch, 'fm_precision'] = dev_fm_precision
        all_epoch_data.at[epoch, 'fm_recall'] = dev_fm_recall

        all_epoch_data.at[epoch, 'saho_f1'] = dev_saho_f1
        all_epoch_data.at[epoch, 'saho_precision'] = dev_saho_precision
        all_epoch_data.at[epoch, 'saho_recall'] = dev_saho_recall  

        all_epoch_data.at[epoch, 'sc_f1'] = dev_sc_f1
        all_epoch_data.at[epoch, 'sc_precision'] = dev_sc_precision
        all_epoch_data.at[epoch, 'sc_recall'] = dev_sc_recall

        #saving model
        if dev_net_f1 > best_net_f1:
            best_net_f1 = dev_net_f1
            best_dev_acc = dev_accuracy
            
            #best_test_acc = test_accuracy
            best_epoch = epoch

            best_ind_f1 = [dev_fm_f1, dev_saho_f1, dev_sc_f1]
            best_ind_precision = [dev_fm_precision, dev_saho_precision, dev_sc_precision]
            best_ind_recall = [dev_fm_recall, dev_saho_recall, dev_saho_recall]

            best_overall_prediction_data = dev_overall_prediction
            
            if model_save_flag:
                os.makedirs(model_save_location, exist_ok=True)
                tokenizer.save_pretrained(model_save_location)
                model.save_pretrained(model_save_location)

        '''if best_tb_acc < test_accuracy_tb:
            best_tb_acc = test_accuracy_tb
            best_tb_epoch = epoch'''

        now = time.time()
        print('BEST ACCURACY --> ', 'DEV:', round(best_dev_acc, 5))
        print('BEST NET F1 --> ', 'DEV:', round(best_net_f1, 5))
        print('TIME PER EPOCH:', (now-start)/60 )
        print()

    return best_overall_prediction_data, best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_net_f1, best_ind_f1, best_ind_precision, best_ind_recall, all_epoch_data





if __name__ == '__main__':
    n_epochs = 1
    models = ['bert-base-uncased', 'roberta-base']
    
    #model saving parameters
    model_save_flag = True
    model_load_flag = False

    #setting up the arrays to save data for all loops, models, and epochs
    
    # accuracy
    all_best_dev_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_test_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_tb_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    
    # epoch
    all_best_epoch = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_tb_epoch = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    
    # factors to calculate final f1 performance metric
    all_best_ind_f1_score = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_ind_precision = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_ind_recall = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    # final f1 performance metric
    all_best_overall_f1_score = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    

    for loop_index in range(1):
        for model_name in models:
            print('Running loop', loop_index)
            print()
            
            model_save_location = '../saved_models_2a/' + model_name + '/' + str(loop_index) + '/' 
            model_load_location = None

            epoch_save_location = '../saved_epoch_2a/' + model_name + '/' + str(loop_index) + '/' 
            epoch_save_name = epoch_save_location + '/epoch_info.tsv'

            result_save_location = '../saved_data_2a/' + model_name + '/' + str(loop_index) + '/'

            unformatted_result_save_location = result_save_location + 'unformatted_result.tsv'
            formatted_result_save_location = result_save_location + 'formatted_result.tsv'

            best_prediction_result, best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_overall_f1_score, best_ind_f1_score, best_ind_precision, best_ind_recall, epoch_data = main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location)


            # Getting accuracy
            all_best_dev_acc.at[loop_index, model_name] = best_dev_acc
            all_best_test_acc.at[loop_index, model_name] = best_test_acc
            all_best_tb_acc.at[loop_index, model_name] = best_tb_acc
            
            # Getting best epoch data
            all_best_epoch.at[loop_index, model_name] = best_epoch
            all_best_tb_epoch.at[loop_index, model_name] = best_tb_epoch

            # Getting best overall f1 score
            all_best_overall_f1_score.at[loop_index, model_name] = best_overall_f1_score
            
            # Getting best individual data (by category)
            all_best_ind_f1_score.at[loop_index, model_name] = best_ind_f1_score
            all_best_ind_precision.at[loop_index, model_name] = best_ind_precision
            all_best_ind_recall.at[loop_index, model_name] = best_ind_recall

            # Get all epoch info 
            os.makedirs(epoch_save_location, exist_ok=True)
            epoch_data.to_csv(epoch_save_name, sep='\t')

            print("\n Prediction results")
            print(best_prediction_result)
            formatted_prediction_result = best_prediction_result.drop(columns=['Orig'])

            os.makedirs(result_save_location, exist_ok=True)
            best_prediction_result.to_csv(unformatted_result_save_location, sep='\t', index=False)
            formatted_prediction_result.to_csv(formatted_result_save_location, sep='\t', index=False)

            print("Result files saved")

    # printing results for analysis
    print("\n All best overall f1 score")
    print(all_best_overall_f1_score)

    print("\n All best dev acc")
    print(all_best_dev_acc)

    print("\n All best f1 score")
    print(all_best_ind_f1_score)

    print("\n All best precision")
    print(all_best_ind_precision)

    print("\n All best recall")
    print(all_best_ind_recall)

    #saving all results into tsv

    os.makedirs('../validating_statistics/', exist_ok=True)
    all_best_overall_f1_score.to_csv('../results/all_best_overall_f1_score.tsv', sep='\t')
    all_best_dev_acc.to_csv('../results/all_best_dev_acc.tsv', sep='\t')
    all_best_ind_f1_score.to_csv('../results/all_best_ind_f1_score.tsv', sep='\t')
    all_best_ind_precision.to_csv('../results/all_best_ind_precision.tsv', sep='\t')
    all_best_ind_recall.to_csv('../results/all_best_ind_recall.tsv', sep='\t')

    print("Everything successfully completed")

    



