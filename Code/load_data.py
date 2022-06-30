from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import pandas as pd
import numpy as np
import torch


class dataset(Dataset):
  def __init__(self, all_data, tokenizer, labels_to_ids, max_len):
        self.len = len(all_data)
        self.data = all_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

  def __getitem__(self, index):
        print("got to get item")
        # step 1: get the sentence and word labels 
        sentence = self.data.at[index, 0]
        print("Sentence: ", sentence)
        #joined_sentnece = ' '.join(sentence)
        input_label = self.data.at[index, 1]
        print("Input label: ", input_label)

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)

        labels = self.labels_to_ids[input_label]

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels)

        return item

  def __len__(self):
        return self.len



def initialize_data(face_masks_tokenizer, stay_at_home_orders_tokenizer, school_closures_tokenizer, initialization_input, input_data, labels_to_ids, shuffle = True):
    max_len, batch_size = initialization_input

    to_split_data_df = pd.DataFrame.from_records(input_data)
    print(to_split_data_df)


    # splitting the data based on class
    face_masks_data = to_split_data_df.loc[to_split_data_df[2] == 'face masks']
    stay_at_home_orders_data = to_split_data_df.loc[to_split_data_df[2] == 'stay at home orders']
    school_closures_data = to_split_data_df.loc[to_split_data_df[2] == 'school closures']
    
    print("\n Face masks data:")
    print(face_masks_data)

    print("\n stay_at_home_orders_data")
    print(stay_at_home_orders_data)

    print("\n school closures data")
    print(school_closures_data)

    # Getting separate datasets
    face_masks_dataset = dataset(face_masks_data, face_masks_tokenizer, labels_to_ids, max_len)
    stay_at_home_orders_dataset = dataset(stay_at_home_orders_data, stay_at_home_orders_tokenizer, labels_to_ids, max_len)
    school_closures_dataset = dataset(school_closures_data, school_closures_tokenizer, labels_to_ids, max_len)
    
    

    params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 4
                }

    # Getting separate dataloaders
    face_masks_loader = DataLoader(face_masks_dataset, **params)
    stay_at_home_orders_loader = DataLoader(stay_at_home_orders_dataset, **params)
    school_closures_loader = DataLoader(school_closures_dataset, **params)
    
    return face_masks_loader, stay_at_home_orders_loader, school_closures_loader




if __name__ == '__main__':
  pass