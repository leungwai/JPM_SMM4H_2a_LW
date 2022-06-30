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

def train(epoch, training_loader, model, optimizer, device, grad_step = 1, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    tr_f1_score, tr_precision, tr_recall = 0, 0, 0

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
    
        # calculating the f1_score for ADE label
        tmp_tr_f1_score = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro")
        tr_f1_score += tmp_tr_f1_score
        # print("tmp f1: ", tmp_tr_f1_score)

        # calculating the precision for ADE label
        tmp_tr_precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro") 
        tr_precision += tmp_tr_precision
        # print("\n tmp precision: ", tmp_tr_precision)

        # calculating the recall for ADE label 
        tmp_tr_recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro")
        tr_recall += tmp_tr_recall
        # print("\n tmp recall: ", tmp_tr_recall)

        # debugging - computing the accuracy report
        # tmp_tr_accuracy_report = classification_report(labels.cpu().numpy(), predictions.cpu().numpy())
        # print("\n Classification Report: \n")
        # print(tmp_tr_accuracy_report)

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

    tr_f1_score = tr_f1_score / nb_tr_steps
    tr_precision = tr_precision / nb_tr_steps
    tr_recall = tr_recall / nb_tr_steps

    return model, tr_f1_score, tr_precision, tr_recall


def testing(model, testing_loader, labels_to_ids, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    eval_f1_score, eval_precision, eval_recall = 0, 0, 0
    
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
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
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

            # calculating the f1_score for ADE label
            tmp_eval_f1_score = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro")
            eval_f1_score += tmp_eval_f1_score
            # print("tmp f1: ", tmp_eval_f1_score)

            # calculating the precision for ADE label
            tmp_eval_precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro")
            eval_precision += tmp_eval_precision
            # print("\n tmp precision: ", tmp_eval_precision)

            # calculating the recall for ADE label 
            tmp_eval_recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro")
            eval_recall += tmp_eval_recall
            # print("\n tmp recall: ", tmp_eval_recall)

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    eval_f1_score = eval_f1_score / nb_eval_steps
    eval_precision = eval_precision / nb_eval_steps
    eval_recall = eval_recall / nb_eval_steps
    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions, eval_accuracy, eval_f1_score, eval_precision, eval_recall


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
        face_masks_tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        face_masks_model = AutoModelForSequenceClassification.from_pretrained(model_load_location)

        stay_at_home_orders_tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        stay_at_home_orders_model = AutoModelForSequenceClassification.from_pretrained(model_load_location)

        school_closures_tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        school_closures_model = AutoModelForSequenceClassification.from_pretrained(model_load_location)
     
    else: 
        face_masks_tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        face_masks_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))

        stay_at_home_orders_tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        stay_at_home_orders_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))

        school_closures_tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        school_closures_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_to_ids))
    
    # Setting up the optimizer for each class
    face_masks_optimizer = torch.optim.Adam(params=face_masks_model.parameters(), lr=learning_rate)
    stay_at_home_orders_optimizer = torch.optim.Adam(params=stay_at_home_orders_model.parameters(), lr=learning_rate)
    school_closures_optimizer = torch.optim.Adam(params=school_closures_model.parameters(), lr=learning_rate)
    
    # Sending each of the model to the GPU
    face_masks_model.to(device)
    stay_at_home_orders_model.to(device)
    school_closures_model

    #Get dataloaders for each separate topic
    train_face_masks_loader, train_stay_at_home_orders_loader, train_school_closures_loader = initialize_data(face_masks_tokenizer, stay_at_home_orders_tokenizer, school_closures_tokenizer, initialization_input, train_data, labels_to_ids, shuffle = True)
    dev_face_masks_loader, dev_stay_at_home_orders_loader, dev_school_closures_loader = initialize_data(face_masks_tokenizer, stay_at_home_orders_tokenizer, school_closures_tokenizer, initialization_input, dev_data, labels_to_ids, shuffle = True)

    print("Got dataloaders working")
    quit()
    #test_loader = initialize_data(tokenizer, initialization_input, test_data, labels_to_ids, shuffle = True)#create test loader

    best_dev_acc = 0
    best_test_acc = 0
    best_epoch = -1
    best_tb_acc = 0
    best_tb_epoch = -1

    best_f1_score = 0
    best_precision = 0
    best_recall = 0

    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train model
        model, tr_f1_score, tr_precision, tr_recall = train(epoch, train_loader, model, optimizer, device, grad_step)
        
        #testing and logging
        labels_dev, predictions_dev, dev_accuracy, dev_f1_score, dev_precision, dev_recall = testing(model, dev_loader, labels_to_ids, device)
        print('DEV ACC:', dev_accuracy)
        print('DEV F1:', dev_f1_score)
        print('DEV PRECISION:', dev_precision)
        print('DEV RECALL:', dev_recall)

        #labels_test, predictions_test, test_accuracy = testing(model, test_loader, labels_to_ids, device)
        #print('TEST ACC:', test_accuracy)

        #saving model
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            
            best_f1_score = dev_f1_score
            best_precision = dev_precision
            best_recall = dev_recall
            
            #best_test_acc = test_accuracy
            best_epoch = epoch
            
            if model_save_flag:
                os.makedirs(model_save_location, exist_ok=True)
                tokenizer.save_pretrained(model_save_location)
                model.save_pretrained(model_save_location)

        '''if best_tb_acc < test_accuracy_tb:
            best_tb_acc = test_accuracy_tb
            best_tb_epoch = epoch'''

        now = time.time()
        print('BEST ACCURACY --> ', 'DEV:', round(best_dev_acc, 5))
        print('BEST F1 --> ', 'DEV:', round(best_f1_score, 5))
        print('BEST PRECISION --> ', 'DEV:', round(best_precision, 5))
        print('BEST RECALL --> ', 'DEV:', round(best_recall, 5))
        print('TIME PER EPOCH:', (now-start)/60 )
        print()

    return best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_f1_score, best_precision, best_recall, 0





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
    all_best_f1_score = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_precision = pd.DataFrame(index=[0,1,2,3,4], columns=models)
    all_best_recall = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    # final f1 performance metric
    all_best_performance = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    # epoch data
    all_epoch_data = pd.DataFrame(index=[0,1,2,3,4], columns=models)

    for model_name in models:

        for loop_index in range(2):
            print('Running loop', loop_index, ': \n')
            model_save_location = '../saved_models_2a/' + model_name + '/' + str(loop_index)
            model_load_location = None

            best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch, best_f1_score, best_precision, best_recall, best_performance, epoch_data = main(n_epochs, model_name, model_save_flag, model_save_location, model_load_flag, model_load_location)

            # accuracy
            all_best_dev_acc.at[loop_index, model_name] = best_dev_acc
            all_best_test_acc.at[loop_index, model_name] = best_test_acc
            all_best_tb_acc.at[loop_index, model_name] = best_tb_acc

            # best epoch data
            all_best_epoch.at[loop_index, model_name] = best_epoch
            all_best_tb_epoch.at[loop_index, model_name] = best_tb_epoch

            # factors to calculate performance metric 
            all_best_f1_score.at[loop_index, model_name] = best_f1_score
            all_best_precision.at[loop_index, model_name] = best_precision
            all_best_recall.at[loop_index, model_name] = best_recall

            # final factors
            all_best_performance.at[loop_index, model_name] = best_recall
            all_epoch_data.at[loop_index, model_name] = best_recall

    # printing results for analysis
    print("\n All best dev acc")
    print(all_best_dev_acc)

    print("\n All best f1 score")
    print(all_best_f1_score)

    print("\n All best precision")
    print(all_best_precision)

    print("\n All best recall")
    print(all_best_recall)

    print("\n All best performance")
    print(all_best_performance)
    
    print("\n All epoch data")
    print(all_epoch_data)



