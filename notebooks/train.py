from fetch_dataset import pennDataset
from model import CompositionalNetwork, Tagger
from typing import Dict
import pickle
import torch
from torch.utils.data import DataLoader

# Step 1: Load the Train and Test datasets:
def collate_function(batch):
    return(batch)
#-------------------------------------------------------------------
print("1. Loading Train/Test/Validation datasets...")
train_ds = DataLoader(dataset=pennDataset('train_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
test_ds = DataLoader(dataset=pennDataset('test_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
valid_ds = DataLoader(dataset=pennDataset('valid_dataset_transformed.json'), shuffle=False, batch_size=1, collate_fn=collate_function)
print("Done! All datasets are loaded!")
print("-"*100)

#-------------------------------------------------------------------
# Step 2: Lookup tables
print("2. Loading lookup tables...")
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
tokens_lkp, tags_lkp, targets_lkp = load_lkp_tables()
print("done!"+"-"*100)
#-------------------------------------------------------------------
# Step 3: Define Hyperparameters:
print("3. Loading hyperparameters...")
OUTPUT_DIM = 10
COMP_EMB_DIM = 20
WORD_EMB_DIM = 20
TAG_EMB_DIM = 5
VOCAB_SIZE = len(tokens_lkp)
N_COMP_NETWORKS = 4
TREE_DEPTH = 2
print("done!")
print("-"*100)
#-------------------------------------------------------------------
# Step 4: Initialize Neural Network models
print("4 Building NN models...")
compositional_model = CompositionalNetwork(output_dim=OUTPUT_DIM
                                    , vocab_size=VOCAB_SIZE)
tagger_model = Tagger(output_dim=OUTPUT_DIM
                    , comp_emb_dim=COMP_EMB_DIM)
print(compositional_model)
print('-'*100)
print(tagger_model)
print('done!')
print('='*100)
#-------------------------------------------------------------------
# Step 5: Function: Model training
def train_loop(
    loss_fn,
    optimizer,
    train_dataset,
    tagger_model: Tagger,
    comp_model: compositional_model,
    # dataloader: DataLoader,
):
    """training loop"""
    # iterate through the dataset
    # each batch is a nested dict
    for batch in train_dataset:
        # for each batch we work through entire tree
        temp_tagger_predictions = dict()
        temp_compositional_output = dict()
        # samples_in_curr_batch = dict()
        for level in range(2, TREE_DEPTH + 1):
            print(f"Length of batch: {len(batch)} and current level: {level}")
            # print(batch)
            print(batch[0][str(level)])
            print('-'*50)
            # for first level, we use POS tags
            # TODO replace tokens with their index -- done already
            if level == 2:
                input_dict = {
                    "token_indices": batch[0]['2']["tokens"],
                    "tag_indices": batch[0]['2']["tags"],
                    "tags": batch[0]['2']["tags"], #original tags before idx look-up
                    "target_indices": batch[0]['2']["targets"],
                    "level": '2',
                    "use_embedding": True,
                }

            # for other levels, we use predicted tags of previous level from the tagger model
            else:
                
                input_dict = {
                    "tokens": temp_compositional_output[level - 1],
                    "tag_indices": temp_tagger_predictions[level - 1],
                    "tags": [
                        comp_model.idx_to_label(idx)
                        for idx in temp_tagger_predictions[level - 1]
                    ],
                    "target_indices": [
                        comp_model.label_to_idx(tag, tag_type="constituents")
                        for tag in batch[level]["targets"]
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

for epoch in range(5):
    print(f"Epoch #: {epoch}")
    # optim = torch.optim.Adam(
    #     params=[tagger_model.parameters(), compositional_model.parameters()]
    # )
    params=list(tagger_model.parameters())+ list(compositional_model.parameters())
    optim = torch.optim.Adam(params)
    loss_fn = torch.nn.CrossEntropyLoss()
    print("5. Starting Model training....")
    train_loop(
        loss_fn=loss_fn,
        train_dataset=train_ds,
        tagger_model=tagger_model,
        comp_model=compositional_model,
        optimizer=optim,
    )
    
# def main():
#     """main function"""
#     # tagger_model = Tagger(output_dim=OUTPUT_DIM, comp_emb_dim=COMP_EMB_DIM)
#     # compositional_model = CompositionalNetwork(
#     #     output_dim=OUTPUT_DIM,
#     #     vocab_size=VOCAB_SIZE,
#     #     word_emd_dim=WORD_EMB_DIM,
#     #     comp_emb_dim=COMP_EMB_DIM,
#     #     n_comp_layers=N_COMP_NETWORKS,
#     # )
#     optim = torch.optim.Adam(
#         params=[tagger_model.parameters(), compositional_model.parameters()]
#     )
#     loss_fn = torch.nn.CrossEntropyLoss()
#     # dataloader = DataLoader(Dataset(), batch_size=1)
#     print("5. Starting Model training....")
#     train_loop(
#         loss_fn=loss_fn,
#         dataloader=train_ds,
#         tagger_model=tagger_model,
#         comp_model=compositional_model,
#         optimizer=optim,
#     )