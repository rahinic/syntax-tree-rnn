import torch
from typing import Dict
class CompositionalNetwork(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        vocab_size: int,
        tag_size: int,
        word_emd_dim: int = 200,
        tag_emb_dim: int = 20,
        comp_emb_dim: int = 200,
        n_comp_layers: int = 4,
    ):
        super(CompositionalNetwork, self).__init__()
        self.word_emb_layer = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=word_emd_dim
        )
        self.tag_emb_layer = torch.nn.Embedding(
            # num_embeddings=output_dim, embedding_dim=tag_emb_dim
            num_embeddings=tag_size, embedding_dim=tag_emb_dim
        )
        self.compositional_layers = {
            k: torch.nn.Linear(
                in_features=(word_emd_dim + tag_emb_dim) * k, out_features=comp_emb_dim
            )
            for k in range(1, n_comp_layers + 1)
        }

    def identify_chunks(self, tags, level):
        """
        identify chunks using BIOES tags. Return a list of <chunk index, length> tuples
        """
        # if level == 1, then each token is a standalone chunk
        if level == 1:
            return [(i, 1) for i in range(len(tags))]
        chunks = list()
        current_chunk = {"start_index": -1, "length": 0}
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
                elif tag.split("-")[0] == "B":
                    current_chunk = {"start_index": i, "length": 1}
                else:
                    current_chunk["length"] += 1
        if current_chunk["start_index"] != -1:
            chunks.append((current_chunk["start_index"], current_chunk["length"]))
        return chunks

    def forward(self, x: Dict):
        """
        :param x: a dict of tokens and tags. Tags are used to identify chunks, which decide which compositional layer to use.
        
        """
        chunks = self.identify_chunks(x["tags"], level=x["level"])
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
                    ],
                    tag_embeddings[
                        chunk_start_index : (chunk_start_index + chunk_length)
                    ],
                ]
            )
            composed_embeddings.append(
                self.compositional_layers[chunk_length](stacked_embeddings)
            )
        return torch.cat(composed_embeddings, dim=0)

#####################################################################################################

class Tagger(torch.nn.Module):
    def __init__(self, output_dim: int, comp_emb_dim: int, rnn_dim: int = 128):
        super(Tagger, self).__init__()
        self.recurrent_layer = torch.nn.LSTM(
            input_size=comp_emb_dim, hidden_size=rnn_dim, batch_first=True
        )
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