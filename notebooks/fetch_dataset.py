"""Pytorch dataset iterator"""

import json
import pickle
import torch
# from torch._C import int64

def fetch_jsonfile(input_dataset_file):
    """returns the .json format of the requested dataset file"""
    filepath = "data/interim/"
    with open(filepath+input_dataset_file,'r') as f:
        contents = json.load(f)

    return contents

def load_lkp_tables():
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


def file_processor(input_dataset):

    """
    Does the following:
        1. Fetch the JSON file of the requested dataset file
        2. Fetch the 3 look-up tables W, T and P
        3. Defines the tokens/tags/targets to idx conversion pipelines
        4. depending on the type of list(subllist, no sublist, mix of both), uses #3 and makes the conversion to idx
        5. idx to tensor conversion 
    """

    dataset_samples = fetch_jsonfile(input_dataset) # Step 1
    token_lkp_tbl, tags_lkp_tbl, tgt_lkp_tbl = load_lkp_tables() # Step 2


    # Step 3
    def sample_pipeline(x):
        
        curr_token_to_idx_list = []
        sub_lst = []
        for lst in x:
            if len(lst) == 1:
                try:
                    # print(f"The token {lst[0]} has the idx: {token_lkp_tbl[lst[0]]}")
                    sub_lst.append([token_lkp_tbl[lst[0]]])
                    # print("SU")
                except TypeError:
                    print(f"Type Error here: {lst[0]}")
                
            else:
                print(f"Big sublist:{lst}")
                sub_lst.append([token_lkp_tbl[tok] for tok in lst])

        curr_token_to_idx_list.append(torch.tensor(sub_lst))
            # print(sub_lst)


        return curr_token_to_idx_list

    def tags_pipeline(x):
        return [tags_lkp_tbl[tag] for tag in x]

    def tgt_pipeline(x):
        return [tgt_lkp_tbl[tgt] for tgt in x]
    
    # (i) Consider one example object from our JSON file
    example = dataset_samples[8]
    max_level = list(example.keys())[-1]
    print(f"One example: {example}")
    print(f"Last level here is : {max_level}")
    
    # (ii) Create a skeletol dict similar to this example dict of JSON obj
    
    dict_as_numbers = {
        level: {"tokens": list(), "tags": list(), "targets": list()}
        for level in range(2, int(max_level)+1)
    }

    # (iii) 
    for level in example.items():
        print(level[1]['tokens'])
        curr_token_list = level[1]['tokens']
        print(sample_pipeline(curr_token_list))

        print('------') 
    # print(dict_as_numbers)

file_processor('test_dataset_transformed.json')