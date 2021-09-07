import json
import pickle

class PennTreebankDictionary():

    def load_json(self, json_file):
        filepath = 'data/interim/'
        with open(filepath+json_file) as f:
            all_samples = json.load(f)
        return all_samples

    def all_tokens_tags_targets(self, input_samples):
        """aggregates unique Tokens | Targets | Tags from each sample"""

        tokens, tags, targets = [],[],[]

        for sample in input_samples:
            # tokens
            all_tokens = [item for sublist in sample["2"]["tokens"] for item in sublist]
            tokens.append(all_tokens)

            #tags and targets
            all_levels = list(sample.keys()) # find out all possible levels
            all_tags, all_targets = [], []
            for level in all_levels:
                all_tags.append(sample[level]["tags"])
                all_targets.append(sample[level]["targets"])

            all_tags_flat = list(set([item for sublist in all_tags for item in sublist]))
            all_targets_flat = list(set([item for sublist in all_targets for item in sublist]))
            
            
            tags.append(all_tags_flat)
            targets.append(all_targets_flat)

        combine_every_sentence_tokens = [item for sublist in tokens for item in sublist]
        combine_every_sentence_tags = [item for sublist in tags for item in sublist]
        combine_every_sentence_targets = [item for sublist in targets for item in sublist]

        return set(combine_every_sentence_tokens), set(combine_every_sentence_tags), set(combine_every_sentence_targets)


lookup = PennTreebankDictionary()

datasets = ['train_dataset_transformed.json','test_dataset_transformed.json','valid_dataset_transformed.json']
datasets_combined = []

for dataset in datasets:
    datasets_combined.append(lookup.load_json(dataset))
datasets_combined_flattened =  [item for sublist in datasets_combined for item in sublist]

tok, tag, target = lookup.all_tokens_tags_targets(datasets_combined_flattened)

print(f"There are in total {len(tok)} tokens, {len(tag)} POS tags and {len(target)}")


############## file exports:
def write_pickle(input_dict,export_filename):
    filepath = 'data/interim/'
    with open(filepath+export_filename, 'wb') as f:
        pickle.dump(input_dict, f)    

############# look-up dictionary construction
tokens_lkp, tags_lkp, targets_lkp = {},{},{}
tokens_lkp_rev, tags_lkp_rev, targets_lkp_rev = {},{},{}

print("Creating the lookup tables and exporting to .pkl files")
for idx, t in enumerate(tok):
    tokens_lkp[t] = idx
    tokens_lkp_rev[idx] = t
write_pickle(input_dict=tokens_lkp,export_filename="tokens_lkp.pkl")
write_pickle(input_dict=tokens_lkp_rev,export_filename="tokens_lkp_rev.pkl")

for idx, p in enumerate(tag):
    tags_lkp[p] = idx
    tags_lkp_rev[idx] = p
write_pickle(input_dict=tags_lkp,export_filename="tags_lkp.pkl")
write_pickle(input_dict=tags_lkp_rev,export_filename="tags_lkp_rev.pkl")

for idx, t2 in enumerate(target):
    targets_lkp[t2] = idx
    targets_lkp_rev[idx] = t2       
write_pickle(input_dict=targets_lkp,export_filename="targets_lkp.pkl")
write_pickle(input_dict=targets_lkp_rev,export_filename="targets_lkp_rev.pkl")
print("done!")
