import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF
from torch.cuda.amp import autocast
import numpy as np

class AutoModelCRF(nn.Module):
    def __init__(self, model_name_or_path, dropout=0.1):
        super(AutoModelCRF, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.num_labels = 2
        self.encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        outputs = self.encoder(**inputs)
        seq_output = outputs[0]
        seq_output = self.dropout(seq_output)
        emission = self.linear(seq_output)
        if labels is None:
            tags = self.crf.decode(emission, attention_mask.byte())
            tags_padded = []
            for idx, sequence in enumerate(tags):
                if len(attention_mask[idx]) > len(sequence):
                    tag_padded = sequence + [sequence[-1]]*(len(attention_mask[idx])-len(sequence))
                else:
                    tag_padded = sequence
                tags_padded.append(tag_padded)
            out = np.array(tags_padded)
            return out
        else:
            crf_loss = -self.crf(emission, labels, mask=attention_mask.byte())
            return crf_loss


        