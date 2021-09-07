"""Pytorch dataset iterator"""

import json
import pickle
from typing import Type
import torch
from torch.utils.data import Dataset, DataLoader

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
        dataset = []
        for sample in dataset_samples:
            max_level = list(sample.keys())[-1]
            # (ii) Create a skeletol dict similar to this example dict of JSON obj

            dict_as_numbers = {
                str(level): {"tokens": list(), "tags": list(), "targets": list()}
                for level in range(2, int(max_level)+1)
            }

            for level in sample.items():

                curr_token_list = level[1]['tokens']
                curr_token_list_as_numbers = sample_to_idx_pipeline(curr_token_list,input_lkp_tbl=token_lkp_tbl)
                dict_as_numbers[level[0]]['tokens'] = curr_token_list_as_numbers
                # print(dict_as_numbers)

                curr_tag_list = level[1]['tags']        
                # print(curr_tag_list)
                curr_tags_list_as_numbers = sample_to_idx_pipeline(curr_tag_list,input_lkp_tbl=tags_lkp_tbl)
                # print(curr_tags_list_as_numbers)
                dict_as_numbers[level[0]]['tags'] = curr_tags_list_as_numbers

                curr_tgt_list = level[1]['targets']        
                curr_tgt_list_as_numbers = sample_to_idx_pipeline(curr_tgt_list,input_lkp_tbl=tgt_lkp_tbl)
                dict_as_numbers[level[0]]['targets'] = curr_tgt_list_as_numbers 

                dataset.append(dict_as_numbers)
            
            return dataset
#--------------------------------------------------------------
    def __init__(self, currDataset=None):

        self.mydataset = currDataset
        # self.vocabulary, self.pos_tags, self.target_tags = self.load_lkp_tables()
        self.all_samples = self.file_processor(input_dataset=self.mydataset)
        # self.samples = self.file_parser(self.all_samples)

    def __len__(self):
        print(len(self.all_samples))

        return len(self.all_samples)

    def __getitem__(self, idx) :

        return self.all_samples[idx]   

valid_ds = DataLoader(dataset=pennDataset('valid_dataset_transformed.json'), batch_size=8, shuffle=False)

for idx, ex in enumerate(valid_ds):
    if idx>0:
        break
    print(len(ex))
    print(ex)
        # # (i) Consider one example object from our JSON file
        # example = dataset_samples[8]
        # max_level = list(example.keys())[-1]
        # print(f"One example: {example}")
        # # print(f"Last level here is : {max_level}")
        
        # # (ii) Create a skeletol dict similar to this example dict of JSON obj

        # dict_as_numbers = {
        #     str(level): {"tokens": list(), "tags": list(), "targets": list()}
        #     for level in range(2, int(max_level)+1)
        # }

        # # (iii) 
        # for level in example.items():

        #     curr_token_list = level[1]['tokens']
        #     curr_token_list_as_numbers = sample_to_idx_pipeline(curr_token_list,input_lkp_tbl=token_lkp_tbl)
        #     dict_as_numbers[level[0]]['tokens'] = curr_token_list_as_numbers
        #     # print(dict_as_numbers)

        #     curr_tag_list = level[1]['tags']        
        #     # print(curr_tag_list)
        #     curr_tags_list_as_numbers = sample_to_idx_pipeline(curr_tag_list,input_lkp_tbl=tags_lkp_tbl)
        #     print(curr_tags_list_as_numbers)
        #     dict_as_numbers[level[0]]['tags'] = curr_tags_list_as_numbers

        #     curr_tgt_list = level[1]['targets']        
        #     curr_tgt_list_as_numbers = sample_to_idx_pipeline(curr_tgt_list,input_lkp_tbl=tgt_lkp_tbl)
        #     dict_as_numbers[level[0]]['targets'] = curr_tgt_list_as_numbers      


        #     print('------') 
        # print(dict_as_numbers)

# file_processor('test_dataset_transformed.json')
