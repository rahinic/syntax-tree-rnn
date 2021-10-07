import torch 
from typing import Dict


# class 1: targets (BIOES prefixed NP/VP/PP/root) phrase identification tagger

class Tagger(torch.nn.Module):
    def __init__(self, output_dim: int, comp_emb_dim: int, rnn_dim: int = 128):
        super(Tagger, self).__init__()
        #layer 1: LSTM
        self.recurrent_layer = torch.nn.LSTM(
            input_size=comp_emb_dim, hidden_size=rnn_dim, batch_first=True
        )
        #layer 2: linear -- should this be a non-linear layer like Denis mentioned?!
        self.output_layer = torch.nn.Linear(
            in_features=rnn_dim, out_features=output_dim
        )

    def forward(self, x):
        """
        forward pass. Input 'x' has shape <batch_size, sequence_length, embeddings>
        """
        rnn_output, _ = self.recurrent_layer(x)
        return self.output_layer(rnn_output)

######################################################################################################

# Class 2: compostion network that helps form representation of combined tokens

class CompositionalNetwork(torch.nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        tag_size: int,
        word_emb_dim: int=200,        
        tag_emb_dim: int=20,
        comp_emb_dim: int=200,
        n_comp_layers: int=4
    ):
        super(CompositionalNetwork, self).__init__()
        #layer 1: word and tag embedding layers
        # (a) Word embedding layer: Dim: tag_size(46349) x tag_emb_dim(200)
        self.word_emb_layer = torch.nn.Embedding(num_embeddings= vocab_size, embedding_dim= word_emb_dim)
        # (b) Tags embedding layer: Dim: tag_size(73) x tag_emb_dim(20)
        self.tag_emb_layer = torch.nn.Embedding(num_embeddings=tag_size, embedding_dim=tag_emb_dim)
        #layer 2: compositional layer: h(M_pow(k)*z)
        self.compositional_layers = {
            k: torch.nn.Linear(in_features=(word_emb_dim+tag_emb_dim) * k,
                                out_features=comp_emb_dim)
            for k in range(1, n_comp_layers + 1) # total levels + 1 (root)
        }

       
###################################################################
    def identify_chunks(self, tags, level):
        """
        identify chunks using BIOES tags. Return a list of <chunk index, length> tuples
        """
        print('-'*50)
        print("Inside _identify_chunk_ function:")
        print(tags)
        print(level)
        
        # if level == 1, then each token is a standalone chunk
        if level == 1:
            return [(i, 1) for i in range(len(tags))]
        chunks = list()
        current_chunk = {"start_index": -1, "length": 0}
        print(f"2. current chunk is {current_chunk}")
        print(f"Starting the sub-process. targets list: {tags}")
        print('-'*50)
        for i, tag in enumerate(tags):
            if tag == "O" or tag.split("-")[0] == "S":
                if current_chunk["start_index"] != -1:
                    chunks.append(
                        (current_chunk["start_index"], current_chunk["length"])
                    )
                chunks.append((i, 1))
                current_chunk = {"start_index": -1, "length": 0}
                
            else:
                # check if current tag starts with 'E'
                if tag.split("-")[0] == "E":
                    if current_chunk["start_index"] == -1:
                        current_chunk["start_index"] = i
                    chunks.append(
                        (current_chunk["start_index"], current_chunk["length"] + 1)
                    )
                    current_chunk = {"start_index": -1, "length": 0}
                    print(f"2.b) chunks when the tag is E: {current_chunk}")
                elif tag.split("-")[0] == "B":
                    
                    current_chunk = {"start_index": i, "length": 1}
                    
                else:
                    current_chunk["length"] += 1
            
        if current_chunk["start_index"] != -1:
            chunks.append((current_chunk["start_index"], current_chunk["length"]))
        return chunks
#######################################################################################
    def forward(self, x: Dict):
        """
        :param x: a dict of tokens and tags. Tags are used to identify chunks, which decide which compositional layer to use.
        
        """
        
        # chunks = self.identify_chunks(x["tags"], level=x["level"]) 
        chunks = self.identify_chunks(x["targets"], level=x["level"]) 
        
        print(f"Current chunks: {chunks}")
        if x["use_embedding"]:
            token_embeddings = self.word_emb_layer(x["token_indices"])
        else:
            token_embeddings = torch.vstack(x["composed_vectors"])
        tag_embeddings = self.tag_emb_layer(x["tag_indices"])
        
        # iterate through chunks, and pass each through appropriate compostional layer
        composed_embeddings = []
        for chunk_start_index, chunk_length in chunks:
            
            stacked_embeddings = torch.hstack(
                [
                    token_embeddings[
                        chunk_start_index : (chunk_start_index + chunk_length)
                    ].squeeze(dim=1),
                    tag_embeddings[
                        chunk_start_index : (chunk_start_index + chunk_length)
                    ],
                ]
            )
            print(f"Intermediate results:{stacked_embeddings}")
            composed_embeddings.append(
                self.compositional_layers[chunk_length](stacked_embeddings)
            )
        # print(composed_embeddings)
        return torch.cat(composed_embeddings, dim=0)
######################################################################################          