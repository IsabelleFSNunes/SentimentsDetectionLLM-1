import torch

class SentimentDataset(Dataset):
    def __init__(self, txt_list, labels, tokenizer, max_lenght):
        self.input_ids= []
        self.attn_masks= []
        map_label= {
            0: 'negative',
            4: 'positive'
        }

        for txt, label in zip( txt_list, labels ):
            preprocessing = f'<startoftext>Tweet: {txt}\Sentiment: {map_label[label]}<endoftext>'
            encoding_dict = tokenizer(preprocessing,
                                      truncation= True,
                                      max_lenght= max_lenght,
                                      padding= "max_lenght")

            self.input_ids.append( torch.tensor( encoding_dict['input_ids'] ) )
            self.attn_masks.append( torch.tensor( encoding_dict['attention_mask']))
            self.labels.append( map_label[label] )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]
    