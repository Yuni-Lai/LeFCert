from transformers import AutoModel
import numpy as np
import torch
from gt import GPS
'''
   copy from graphclip.py in GraphCLIP, but only use encode_graph actually !!!
'''


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



text_ids = {
    'tiny': 'sentence-transformers/all-MiniLM-L6-v2',
    'sbert':  'sentence-transformers/multi-qa-distilbert-cos-v1', #'sentence-transformers/all-MiniLM-L6-v2', #'sentence-transformers/multi-qa-distilbert-cos-v1',
    'e5': 'intfloat/e5-base-v2',
    'deberta': 'microsoft/deberta-v3-base',
}



class Config:
    def __init__(self):
        self.prefix_projection=True
        self.pre_seq_len=10
        self.num_hidden_layers=6
        self.prefix_hidden_size=384
        self.hidden_size=384

config = Config()

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class GraphCLIP(torch.nn.Module):
    def __init__(self, graph_input_dim, graph_hid_dim, graph_num_layer, attn_kwargs, text_model='tiny'):
        super().__init__()
        self.graph_model = GPS(in_dim=graph_input_dim, channels=graph_hid_dim, out_dim=graph_hid_dim, 
                               pe_dim=8, num_layers=graph_num_layer, attn_type='multihead', attn_kwargs=attn_kwargs)
        self.text_model_type = text_model
        text_id = text_ids[text_model]
        text_model = AutoModel.from_pretrained(text_id)
        self.text_model = text_model
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #most important
    def encode_graph(self, batch):
        graph_embs, center_embs = self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index)
        # node_emb=self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index)
        return graph_embs #有修改，only return node embedding

    def encode_text(self, input_ids, token_type_ids, attention_mask):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)
        return text_embs

    def forward(self, batch_g, batch_t):
        graph_features, c_features = self.encode_graph(batch_g)
        text_features = self.encode_text(**batch_t)

        # normalized features
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_graph = logit_scale * graph_features @ text_features.t()
        logits_per_text = logits_per_graph.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_graph, logits_per_text
    
    def freeze_text(self):
        for k, v in self.text_model.named_parameters():
            v.requires_grad = False