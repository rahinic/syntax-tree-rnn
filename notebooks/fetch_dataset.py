"""Pytorch dataset iterator"""

import json
import pickle
from typing import Type
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

class pennDataset(Dataset):
    
    def fetch_jsonfile(self, input_dataset_file):
        """returns the .json format of the requested dataset file"""
        filepath = "data/interim/"
        with open(filepath+input_dataset_file,'r') as f:
            contents = json.load(f)

        return contents

    def load_lkp_tables(self):
        """returns W, T and P look-up tables"""
        print("loading dictionaries....")
        filepath = "data/interim/"
        lkp_tbl_lst = ['tokens_lkp.pkl','tags_lkp.pkl','targets_lkp.pkl']

        def pkl_load(input):
            f = open(input,'rb')
            return pickle.load(f)
        
        tokens_lkp = pkl_load(filepath+lkp_tbl_lst[0])
        tags_lkp = pkl_load(filepath+lkp_tbl_lst[1])
        targets_lkp = pkl_load(filepath+lkp_tbl_lst[2])

        print("done")

        return tokens_lkp, tags_lkp, targets_lkp


    def file_processor(self,input_dataset):

        """
        Does the following:
            1. Fetch the JSON file of the requested dataset file
            2. Fetch the 3 look-up tables W, T and P
            3. Defines the tokens/tags/targets to idx conversion pipelines
            4. depending on the type of list(subllist, no sublist, mix of both), uses #3 and makes the conversion to idx
            5. idx to tensor conversion 
        """

        dataset_samples = self.fetch_jsonfile(input_dataset) # Step 1
        token_lkp_tbl, tags_lkp_tbl, tgt_lkp_tbl = self.load_lkp_tables() # Step 2


        # Step 3
        def sample_to_idx_pipeline(input_sample_x, input_lkp_tbl):
            curr_x_to_idx_list, sub_lst, sub_lst_new = [], [], []

            for lst in input_sample_x:
                
                if len(lst)==1:
                    try:
                        sub_lst.append([input_lkp_tbl[lst[0]]])
                    except TypeError:
                        print(f"Type Error here: {lst[0]}")
                else:
                    try:
                        sub_lst.append([input_lkp_tbl[item] for item in lst])                    
                    except KeyError:
                        sub_lst.append(input_lkp_tbl[lst])
            curr_x_to_idx_list.append(sub_lst)

            return curr_x_to_idx_list[0]
        
        def cleaned_list(dirty_indexed_list):
            """"To clean the pipeline result of POS and Target tags resultset"""
            cleaned = []
            for item in dirty_indexed_list:
                if str(type(item))=="<class 'list'>":
                    cleaned.append(item[0]) 
                else:
                    cleaned.append(item)
            return cleaned

        dataset = []
        
        for idx,sample in enumerate(dataset_samples):
            max_level = list(sample.keys())[-1]
            # (ii) Create a skeletol dict similar to this example dict of JSON obj
            
            dict_as_numbers = {
                str(level): {"tokens": list(), "tags": list(), "targets": list()}
                for level in range(2, int(max_level)+1)
                # for level in range(2, 29+1)
            }
            # if int(max_level) > 30:
            #     continue

            for level in sample.items():

                curr_token_list = level[1]['tokens']
                curr_token_list_as_numbers = sample_to_idx_pipeline(curr_token_list,input_lkp_tbl=token_lkp_tbl)
                dict_as_numbers[level[0]]['tokens'] = curr_token_list_as_numbers
                # dict_as_numbers[level[0]]['tokens'] = torch.tensor(curr_token_list_as_numbers)
                

                curr_tag_list = level[1]['tags']
                curr_tags_list_as_numbers = sample_to_idx_pipeline(curr_tag_list,input_lkp_tbl=tags_lkp_tbl)
                curr_tags_list_as_numbers_clean = cleaned_list(curr_tags_list_as_numbers)
                dict_as_numbers[level[0]]['tags'] = curr_tags_list_as_numbers_clean
                # dict_as_numbers[level[0]]['tags'] = torch.tensor(curr_tags_list_as_numbers_clean)

                curr_tgt_list = level[1]['targets']        
                curr_tgt_list_as_numbers = sample_to_idx_pipeline(curr_tgt_list,input_lkp_tbl=tgt_lkp_tbl)
                curr_tgt_list_as_numbers_clean = cleaned_list(curr_tgt_list_as_numbers)
                dict_as_numbers[level[0]]['targets'] = curr_tgt_list_as_numbers_clean
                # dict_as_numbers[level[0]]['targets'] = torch.tensor(curr_tgt_list_as_numbers_clean)

            dataset.append(dict_as_numbers)
        
            
        return dataset
#--------------------------------------------------------------
# test = pennDataset()
# all_samples = test.file_processor(input_dataset='valid_dataset_transformed.json')
# # print(len(all_samples))
# print(all_samples[31])

    def __init__(self, currDataset=None):

        self.mydataset = currDataset
        self.all_samples = self.file_processor(input_dataset=self.mydataset)
        # print(self.all_samples[31])

    def __len__(self):
        # print(len(self.all_samples))

        return len(self.all_samples)

    def __getitem__(self, idx) :
        
        return self.all_samples[idx]   

def collate_function(batch):
    return(batch)

valid_ds = DataLoader(dataset=pennDataset('valid_dataset_transformed.json'), shuffle=False, batch_size=8, collate_fn=collate_function)


for idx,ex in enumerate(valid_ds):
    print(idx)
    print(ex[2])
    print('-'*100)
    if idx>1:
        break

