from fetch_dataset import pennDataset
from torch.utils.data import DataLoader
from model import CompositionalNetwork, Tagger
import torch
import pickle
#----------------------------------------------------------------------------
print("loading train/test/validation datasets...")
def collate_function(batch):
    return(batch)
train_ds = DataLoader(dataset=pennDataset('train_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
test_ds = DataLoader(dataset=pennDataset('test_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
valid_ds = DataLoader(dataset=pennDataset('valid_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
print("done")
#----------------------------------------------------------------------------
print("loading the 3 look-up tables...")

def load_lkp_tables():
        """returns W, T and P look-up tables"""
        print("loading dictionaries....")
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
print("done!")
print("-"*100)   
#---------------------------------------------------------------------------
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

#----------------------------------------------------------------------------
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
#----------------------------------------------------------------------------
print("defining x to idx pipelines...") 
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
print("done.!")
#-----------------------------------------------------------------------------
# Step 4: Initialize Neural Network models
print("4 Building NN models...")
compositional_model = CompositionalNetwork(output_dim=OUTPUT_DIM
                                    , vocab_size=VOCAB_SIZE
                                    , tag_size= TAG_SIZE)
tagger_model = Tagger(output_dim=OUTPUT_DIM
                    , comp_emb_dim=COMP_EMB_DIM)
print(compositional_model)
print('-'*100)
print(tagger_model)
print('done!')
print('='*100)
#-------------------------------------------------------------------
def train_loop(
    loss_fn,
    optimizer,
    tagger_model: Tagger,
    comp_model: CompositionalNetwork,
    train_dataset,
):
    """training loop"""
    # iterate through the dataset
    # each batch is a nested dict
    for batch in train_dataset:
        # for each batch we work through entire tree
        temp_tagger_predictions = dict()
        temp_compositional_output = dict()

        for level in range(2, TREE_DEPTH + 1):
            # for first level, we use POS tags
            # TODO replace tokens with their index
            if level == 2:
                input_dict = {
                    # "token_indices": [token_to_idx(token) for token in batch[0][str(level)]["tokens"]],
                    
                    # "tag_indices": [
                    #     label_to_idx(tag, tag_type="pos")
                    #     for tag in batch[0][str(level)]["tags"]
                    # ],
                    # "target_indices": [
                    #     label_to_idx(tag, tag_type="constituents")
                    #     for tag in batch[0][str(level)]["targets"]
                    # ],
                    "token_indices": sample_to_idx_pipeline(batch[0][str(level)]["tokens"],tokens_lkp),
                    "tag_indices": sample_to_idx_pipeline(batch[0][str(level)]["tags"], tags_lkp),
                    "tags": batch[0][str(level)]["tags"],
                    "target_indices": sample_to_idx_pipeline(batch[0][str(level)]["targets"], targets_lkp),
                    "level": str(level),
                    "use_embedding": True,
                }

            # for other levels, we use predicted tags of previous level from the tagger model
            else:
                input_dict = {
                    "tokens": temp_compositional_output[level - 1],
                    "tag_indices": temp_tagger_predictions[level - 1],
                    "tags": [
                        idx_to_label(idx)
                        for idx in temp_tagger_predictions[level - 1]
                    ],
                    "target_indices": [
                        label_to_idx(tag, tag_type="constituents")
                        for tag in batch[0][str(level)]["targets"]
                    ],
                    "level": level,
                    "use_embedding": False,
                }
            
            composed_output = comp_model(input_dict)
            tagger_output = torch.nn.LogSoftmax(tagger_model(composed_output))
            # TODO do inverse lookup for predictions to get text label from their indices, before storing them in temp_tagger_predictions
            optimizer.zero_grad()
            loss = loss_fn(tagger_output, torch.tensor(batch[level]["target_indices"]))
            loss.backward()
            optimizer.step()

            # store predictions of current level, for use in next level
            temp_tagger_predictions[level] = tagger_output
            temp_compositional_output[level] = composed_output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params=list(tagger_model.parameters())+ list(compositional_model.parameters())
optim = torch.optim.Adam(params)
loss_fn = torch.nn.CrossEntropyLoss()
print("5. Starting Model training....")
for epoch in range(5):
    print(f"Epoch #: {epoch}")
    
    
    train_loop(
        loss_fn=loss_fn,
        train_dataset=train_ds,
        tagger_model=tagger_model,
        comp_model=compositional_model,
        optimizer=optim,
    )