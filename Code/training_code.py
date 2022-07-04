import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from load_data import initialize_data
from reading_datasets import read_task
from labels_to_ids import task7_labels_to_ids
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Global Varibles
FM = 0
SAHO = 1
SC = 2

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

    return model


def testing(model, testing_loader, labels_to_ids, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    eval_f1_score, eval_precision, eval_recall = 0, 0, 0
    
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    eval_tweet_ids, eval_topics, eval_premises, eval_orig_sentences = [], [], [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    overall_prediction_data = []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            tweet_ids = batch['tweet_id']
            topics = batch['topic']
            premises = batch['premise']
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
            eval_premises.extend(premises)
            eval_orig_sentences.extend(orig_sentences)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

            # calculating the f1_score for ADE label
            tmp_eval_f1_score = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0,1], average='macro')
            eval_f1_score += tmp_eval_f1_score

            # print("tmp f1: ", tmp_eval_f1_score)
            # calculating the precision for ADE label
            tmp_eval_precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0,1], average='macro')
            eval_precision += tmp_eval_precision
            # print("\n tmp precision: ", tmp_eval_precision)

            # calculating the recall for ADE label 
            tmp_eval_recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[0,1], average='macro')
            eval_recall += tmp_eval_recall
            # print("\n tmp recall: ", tmp_eval_recall)

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    # print("\n labels:")
    # print(labels)

    # print("\n predictions")
    # print(predictions)

    # print("\n eval_tweet_id")
    # print(eval_tweet_ids)

    # print("\n eval_premises")
    # print(eval_premises)

    # print("\n eval_orig_sentences")
    # print(eval_orig_sentences)
    
    overall_prediction_data = pd.DataFrame(zip(eval_tweet_ids, eval_orig_sentences, eval_topics, eval_premises, labels, predictions), columns=['id', 'text', 'Claim', 'Premise', 'Orig', 'Stance'])


    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    eval_f1_score = eval_f1_score / nb_eval_steps
    eval_precision = eval_precision / nb_eval_steps
    eval_recall = eval_recall / nb_eval_steps
    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return overall_prediction_data, labels, predictions, eval_accuracy, eval_f1_score, eval_precision, eval_recall


def main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location):
    #Initialization training parameters
    max_len = 256
    batch_size = 16
    grad_step = [1, 1, 1]
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

    tokenizer = []
    model = []
    if model_load_flag:
        fm_tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        fm_model = AutoModelForSequenceClassification.from_pretrained(model_load_location)

        saho_tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        saho_model = AutoModelForSequenceClassification.from_pretrained(model_load_location)

        sc_tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        sc_model = AutoModelForSequenceClassification.from_pretrained(model_load_location)
        
        tokenizer = [fm_tokenizer, saho_tokenizer, sc_tokenizer]
        model = [fm_model, saho_model, sc_model]
     
    else: 
        fm_tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        fm_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))

        saho_tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        saho_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))

        sc_tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        sc_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))

        tokenizer = [fm_tokenizer, saho_tokenizer, sc_tokenizer]
        model = [fm_model, saho_model, sc_model]
    
    # Setting up the optimizer for each class
    fm_optimizer = torch.optim.Adam(params=model[FM].parameters(), lr=learning_rate)
    saho_optimizer = torch.optim.Adam(params=model[SAHO].parameters(), lr=learning_rate)
    sc_optimizer = torch.optim.Adam(params=model[SC].parameters(), lr=learning_rate)

    optimizer = [fm_optimizer, saho_optimizer, sc_optimizer]
    
    # Sending each of the model to the GPU
    model[FM].to(device)
    model[SAHO].to(device)
    model[SC].to(device)

    #Get dataloaders for each separate topic
    train_loader = initialize_data(tokenizer, initialization_input, train_data, labels_to_ids, shuffle = True)
    dev_loader = initialize_data(tokenizer, initialization_input, dev_data, labels_to_ids, shuffle = True)

    #test_loader = initialize_data(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = True)#create test loader

    best_dev_predictions = []
    best_test_acc = 0
    best_epoch = -1
    best_tb_acc = 0
    best_tb_epoch = -1

    best_overall_prediction_data = []
    best_overall_f1_score = 0
    best_ind_dev_acc = [0,0,0]
    best_ind_f1_score = [0,0,0]
    best_ind_precision = [0,0,0]
    best_ind_recall = [0,0,0]

    all_epoch_data = pd.DataFrame(index=[0,1,2,3,4,5,6,7,8,9], columns=['overall_f1', 'fm_accuracy', 'fm_f1', 'fm_precision', 'fm_recall', 'saho_accuracy', 'saho_f1', 'saho_precision', 'saho_recall', 'sc_accuracy', 'sc_f1', 'sc_precision', 'sc_recall'])

    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train face_masks model
        print("Training fm model")
        model[FM] = train(epoch, train_loader[FM], model[FM], optimizer[FM], device, grad_step[FM])
        print("Training saho model")
        model[SAHO] = train(epoch, train_loader[SAHO], model[SAHO], optimizer[SAHO], device, grad_step[SAHO])
        print("Training sc model")
        model[SC] = train(epoch, train_loader[SC], model[SC], optimizer[SC], device, grad_step[SC])


        #testing and logging
        print("Testing fm model")
        fm_overall_prediction_data, fm_dev_labels, fm_dev_predictions, fm_dev_accuracy, fm_dev_f1_score, fm_dev_precision, fm_dev_recall = testing(model[FM], dev_loader[FM], labels_to_ids, device)
        print('fm DEV ACC:', fm_dev_accuracy)
        print('fm DEV F1:', fm_dev_f1_score)
        print('fm DEV PRECISION:', fm_dev_precision)
        print('fm DEV RECALL:', fm_dev_recall)

        print("Testing saho model")
        saho_overall_prediction_data, saho_dev_labels, saho_dev_predictions, saho_dev_accuracy, saho_dev_f1_score, saho_dev_precision, saho_dev_recall = testing(model[SAHO], dev_loader[SAHO], labels_to_ids, device)
        print('saho DEV ACC:', saho_dev_accuracy)
        print('saho DEV F1:', saho_dev_f1_score)
        print('saho DEV PRECISION:', saho_dev_precision)
        print('saho DEV RECALL:', saho_dev_recall)

        print("Testing sc model")
        sc_overall_prediction_data, sc_dev_labels, sc_dev_predictions, sc_dev_accuracy, sc_dev_f1_score, sc_dev_precision, sc_dev_recall = testing(model[SC], dev_loader[SC], labels_to_ids, device)
        print('sc DEV ACC:', sc_dev_accuracy)
        print('sc DEV F1:', sc_dev_f1_score)
        print('sc DEV PRECISION:', sc_dev_precision)
        print('sc DEV RECALL:', sc_dev_recall)

        #labels_test, predictions_test, test_accuracy = testing(model, test_loader, labels_to_ids, device)
        #print('TEST ACC:', test_accuracy)
        overall_dev_prediction_data = pd.concat([fm_overall_prediction_data, saho_overall_prediction_data, sc_overall_prediction_data])
        
        overall_dev_f1_score = (1/3)*(fm_dev_f1_score + saho_dev_f1_score + sc_dev_f1_score)
        print('\n DEV OVERALL F1 SCORE:', overall_dev_f1_score)
        
        all_epoch_data.at[epoch, 'overall_f1'] = overall_dev_f1_score

        all_epoch_data.at[epoch, 'fm_accuracy'] = fm_dev_accuracy
        all_epoch_data.at[epoch, 'fm_f1'] = fm_dev_f1_score
        all_epoch_data.at[epoch, 'fm_precision'] = fm_dev_precision
        all_epoch_data.at[epoch, 'fm_recall'] = fm_dev_recall

        all_epoch_data.at[epoch, 'saho_accuracy'] = saho_dev_accuracy
        all_epoch_data.at[epoch, 'saho_f1'] = saho_dev_f1_score
        all_epoch_data.at[epoch, 'saho_precision'] = saho_dev_precision
        all_epoch_data.at[epoch, 'saho_recall'] = saho_dev_recall

        all_epoch_data.at[epoch, 'sc_accuracy'] = sc_dev_accuracy
        all_epoch_data.at[epoch, 'sc_f1'] = sc_dev_f1_score
        all_epoch_data.at[epoch, 'sc_precision'] = sc_dev_precision
        all_epoch_data.at[epoch, 'sc_recall'] = sc_dev_recall        

        #saving model
        if overall_dev_f1_score > best_overall_f1_score:
            best_overall_f1_score = overall_dev_f1_score

            best_ind_dev_acc[FM] = fm_dev_accuracy
            best_ind_f1_score[FM] = fm_dev_f1_score
            best_ind_precision[FM] = fm_dev_precision
            best_ind_recall[FM] = fm_dev_recall

            best_ind_dev_acc[SAHO] = saho_dev_accuracy
            best_ind_f1_score[SAHO] = saho_dev_f1_score
            best_ind_precision[SAHO] = saho_dev_precision
            best_ind_recall[SAHO] = saho_dev_recall

            best_ind_dev_acc[SC] = sc_dev_accuracy
            best_ind_f1_score[SC] = sc_dev_f1_score
            best_ind_precision[SC] = sc_dev_precision
            best_ind_recall[SC] = sc_dev_recall
            
            best_overall_prediction_data = overall_dev_prediction_data

            #best_test_acc = test_accuracy
            best_epoch = epoch
            
            #saving the best predictions so far
            best_dev_predictions.clear()
            best_dev_predictions.append(fm_dev_predictions)
            best_dev_predictions.append(saho_dev_predictions)
            best_dev_predictions.append(sc_dev_predictions)

            if model_save_flag:
                for i in range(3):
                    os.makedirs(model_save_location[i], exist_ok=True)
                    tokenizer[i].save_pretrained(model_save_location[i])
                    model[i].save_pretrained(model_save_location[i])

        '''if best_tb_acc < test_accuracy_tb:
            best_tb_acc = test_accuracy_tb
            best_tb_epoch = epoch'''

        now = time.time()
        print('BEST ACCURACY [FM] --> ', 'DEV:', round(best_ind_dev_acc[FM], 5))
        print('BEST ACCURACY [SAHO] --> ', 'DEV:', round(best_ind_dev_acc[SAHO], 5))
        print('BEST ACCURACY [SC] --> ', 'DEV:', round(best_ind_dev_acc[SC], 5))
        print('BEST F1 --> ', 'DEV:', round(best_overall_f1_score, 5))
        # print('BEST PRECISION --> ', 'DEV:', round(best_precision, 5))
        # print('BEST RECALL --> ', 'DEV:', round(best_recall, 5))
        print('TIME PER EPOCH:', (now-start)/60 )
        print()

    return best_overall_prediction_data, best_ind_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_overall_f1_score, best_ind_f1_score, best_ind_precision, best_ind_recall, all_epoch_data





if __name__ == '__main__':
    n_epochs = 1
    models = ['bert-base-uncased', 'roberta-base']
    
    #model saving parameters
    model_save_flag = True
    model_load_flag = False

    #setting up the arrays to save data for all loops, models, and epochs

    # accuracy
    all_best_ind_dev_acc = pd.DataFrame(index=[0,1,2,3,4], columns=models)
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

    # epoch data
    all_epoch_data = pd.DataFrame(index=[0,1,2,3,4], columns=models)



    for loop_index in range(1):
        for model_name in models:

            print('Running loop', loop_index, ': \n')
            fm_model_save_location = '../saved_models_2a/' + model_name + '/' + str(loop_index) + '/face_masks/' 
            saho_model_save_location = '../saved_models_2a/' + model_name + '/' + str(loop_index) + '/stay_at_home_orders/' 
            sc_model_save_location = '../saved_models_2a/' + model_name + '/' + str(loop_index) + '/school_closures/' 
            model_save_location = [fm_model_save_location, saho_model_save_location, sc_model_save_location]

            result_save_location = '../saved_data_2a/' + model_name + '/' + str(loop_index) + '/'
            unformatted_result_save_location = '../saved_data_2a/' + model_name + '/' + str(loop_index) + '/unformatted_prediction_results.tsv' 
            formatted_result_save_location = '../saved_data_2a/' + model_name + '/' + str(loop_index) + '/formatted_prediction_results.tsv' 
            model_load_location = None

            prediction_result_round, best_ind_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_overall_f1_score, best_ind_f1_score, best_ind_precision, best_ind_recall, epoch_data = main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location)

            # accuracy
            all_best_ind_dev_acc.at[loop_index, model_name] = best_ind_dev_acc
            all_best_test_acc.at[loop_index, model_name] = best_test_acc
            all_best_tb_acc.at[loop_index, model_name] = best_tb_acc

            # best epoch data
            all_best_epoch.at[loop_index, model_name] = best_epoch
            all_best_tb_epoch.at[loop_index, model_name] = best_tb_epoch

            # factors to calculate performance metric 
            all_best_ind_f1_score.at[loop_index, model_name] = best_ind_f1_score
            all_best_ind_precision.at[loop_index, model_name] = best_ind_precision
            all_best_ind_recall.at[loop_index, model_name] = best_ind_recall

            # final factors
            all_best_overall_f1_score.at[loop_index, model_name] = best_overall_f1_score
            all_epoch_data.at[loop_index, model_name] = epoch_data

            print("\n Prediction results")
            print(prediction_result_round)
            formatted_prediction_results = prediction_result_round.drop(columns=['Premise', 'Orig'])

            os.makedirs(result_save_location, exist_ok=True)
            prediction_result_round.to_csv(unformatted_result_save_location, sep='\t', index=False)
            formatted_prediction_results.to_csv(formatted_result_save_location, sep='\t', index=False)

            print("Quit after printing predictions and saving it to file")
            quit()

    # printing results for analysis
    print("\n All best overall f1 score")
    print(all_best_overall_f1_score)

    print("\n All best ind dev acc")
    print(all_best_ind_dev_acc)

    print("\n All best f1 score")
    print(all_best_ind_f1_score)

    print("\n All best precision")
    print(all_best_ind_precision)

    print("\n All best recall")
    print(all_best_ind_recall)

    print("\n All epoch data")
    print(all_epoch_data)

    #saving all results into tsv
    all_best_overall_f1_score.to_csv('results/all_best_overall_f1_score.tsv', sep='\t')
    all_best_ind_dev_acc.to_csv('results/all_best_ind_dev_acc.tsv', sep='\t')
    all_best_ind_f1_score.to_csv('results/all_best_ind_f1_score.tsv', sep='\t')
    all_best_ind_precision.to_csv('results/all_best_ind_precision.tsv', sep='\t')
    all_best_ind_recall.to_csv('results/all_best_ind_recall.tsv', sep='\t')
    all_epoch_data.to_csv('results/all_epoch_data.tsv', sep='\t')



