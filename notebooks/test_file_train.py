from fetch_dataset import pennDataset
from torch.utils.data import DataLoader
from test_file_model import CompositionalNetwork, Tagger
import torch
import pickle
###############################################################################
print("="*100)
print("="*100)
print("1. loading train/test/validation datasets...")
def collate_function(batch):
    return(batch)
# train_ds = DataLoader(dataset=pennDataset('train_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
# test_ds = DataLoader(dataset=pennDataset('test_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
valid_ds = DataLoader(dataset=pennDataset('valid_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
print("done")
print("-"*100)
###############################################################################
print("loading the 3 look-up tables...")

def load_lkp_tables():
        """returns W, T and P look-up tables"""
        print("2. loading dictionaries....")
        filepath = "data/interim/"
        lkp_tbl_lst = ['tokens_lkp.pkl','tags_lkp.pkl','targets_lkp.pkl','tokens_lkp_rev.pkl']

        def pkl_load(input):
            f = open(input,'rb')
            return pickle.load(f)
        
        tokens_lkp = pkl_load(filepath+lkp_tbl_lst[0])
        tags_lkp = pkl_load(filepath+lkp_tbl_lst[1])
        targets_lkp = pkl_load(filepath+lkp_tbl_lst[2])
        tokens_lkp_rev = pkl_load(filepath+lkp_tbl_lst[3])

        print("done")

        return tokens_lkp, tags_lkp, targets_lkp, tokens_lkp_rev
tokens_lkp, tags_lkp, targets_lkp, tokens_lkp_rev = load_lkp_tables()
print(f"Size of look-up tables:\n(a)Tokens:{len(tokens_lkp)}\n(b)Tags:{len(tags_lkp)}\n(c)Targets:{len(targets_lkp)}")
# print("done!")
print("-"*100)   
###############################################################################
# Step 3: Define Hyperparameters:
print("3. Loading hyperparameters...")
OUTPUT_DIM = 10
COMP_EMB_DIM = 20
WORD_EMB_DIM = 20
TAG_EMB_DIM = 5
VOCAB_SIZE = len(tokens_lkp)+1
TAG_SIZE = len(tags_lkp)+1
N_COMP_NETWORKS = 4
TREE_DEPTH = 2
print("done!")
print("-"*100)

###############################################################################
def sample_to_idx_pipeline(input_sample_x, input_lkp_tbl):
    curr_x_to_idx_list, sub_lst = [], []

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

    try:
        return torch.tensor(curr_x_to_idx_list[0])
    except TypeError:
        flat_list = []
        for item in curr_x_to_idx_list[0]:
            # print(type(item))
            if str(type(item)) != "<class 'int'>":
                flat_list.append(item[0])
            else:
                flat_list.append(item)
        # print(flat_list)
        return torch.tensor(flat_list)
        # print(curr_x_to_idx_list[0])
        # return torch.tensor(curr_x_to_idx_list)
###############################################################################
print("4. Defining x to idx pipelines...") 
def idx_to_label(idx, tag_type):
    if tag_type == "pos":
            return tags_lkp[idx]
    return targets_lkp[idx]

def label_to_idx(label, tag_type):
    if tag_type == "pos":
        return tags_lkp.index(label)
    return targets_lkp.index(label)
    
def idx_to_token(idx):
    # return tokens_lkp[idx]
    return tokens_lkp_rev[idx]
    
def token_to_idx(token):
    return tokens_lkp[token]
    # return tokens_lkp.index(token)
print("done!")
print("="*100)
###############################################################################
# Step 5: Initialize Neural Network models
print("5. Building NN models...")
compositional_model = CompositionalNetwork(vocab_size=VOCAB_SIZE, tag_size= TAG_SIZE)
tagger_model = Tagger(output_dim=OUTPUT_DIM, comp_emb_dim=COMP_EMB_DIM)
print(compositional_model)
print('-'*100)
print(tagger_model)
print('done!')
print('='*100)
###############################################################################
# Step 6: Let us validate the x to idx transformation of one example
def fetch_one_example():

    for idx, sample in enumerate(valid_ds):
        if idx>30:
            break
    return sample
example = fetch_one_example()
print(example)
print("="*100)

for level in range(2,TREE_DEPTH+1):
    
    if level == 2:
                input_dict = {
                    "token_indices": sample_to_idx_pipeline(example[0][str(level)]["tokens"],tokens_lkp),
                    "tag_indices": sample_to_idx_pipeline(example[0][str(level)]["tags"], tags_lkp),
                    "targets": example[0][str(level)]["targets"], #change to "tags"
                    "target_indices": sample_to_idx_pipeline(example[0][str(level)]["targets"], targets_lkp),
                    "level": str(level),
                    "use_embedding": True,
                }
    # below is not available in the first iteration as the values are not computed and 
    # stored into the variables `temp_tagger_predictions` and `temp_compositional_output` yet
    
    # else: -- comback to this later

    composed_output = compositional_model(input_dict)
    print(composed_output)