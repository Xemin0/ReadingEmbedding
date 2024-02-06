import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

'''
Transformer section
'''

class AttentionMatrix(nn.Module):
    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def forward(self, inputs):
        '''
        Computes Attention Given key and query matrices.

        Input:
            - K: [batch_size x n_words_keys x embedding_size]
            - Q: [batch_size x n_words_queries x embedding_size]
        Output:
            - Attention Matrix
        '''
        K, Q = inputs
        n_words_queries = Q.shape[1] # window size of queries
        n_words_keys = K.shape[1] # window size of keys
        device = K.device

        ## Fill triangle below diagonal of matrix with negative infinity and top part with 0.
        ## This helps to avoid over-contribution, since adjacency matrix is symmetric across diagonal. 
        ## Tile this upward to be compatible with addition against computed attention scores.

        mask_vals = np.triu(np.ones((n_words_queries)) * np.NINF, k = 1)
        mask = torch.from_numpy(mask_vals).float().to(device)
        atten_mask = torch.tile(torch.reshape(mask,
                                             [-1, n_words_queries, n_words_keys]),
                                             [K.shape[0], 1, 1]) #### Repeat mask along the first dimension / batch
        '''
        Self Attention
        1. compute attention weights using queries and key matrices 
               - if use_mask==True, then make sure to add the attention mask before softmax
        2. return the attention matrix
        
        - Mask: [batch_size x window_size_queries x window_size_keys]
        '''
        weights = torch.einsum('bqe,bke->bqk', Q, K)
        weights /= torch.math.sqrt(float(n_words_keys)) ####

        if self.use_mask:
            weights = torch.add(weights, atten_mask)
        return F.softmax(weights, dim = -1)

class AttentionHead(nn.Module):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize K, V, Q --- Xavier
        scale = 1.0 / torch.math.sqrt( float(input_size + output_size))
        self.Wk, self.Wv, self.Wq = [nn.Parameter(torch.randn(input_size, output_size) * scale) for _ in range(3)]

        # Attention Matrix Layer
        self.atten_matrix = AttentionMatrix(use_mask = self.use_mask)

    def forward(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        '''
        Input:
            - inputs_for_keys: [batch_size x KEY_WINDOW_SIZE x input_size ]
            - inputs_for_values: [batch_size x KEY_WINDOW_SIZE x input_size ]
            - param inputs_for_queries: [batch_size x QUERY_WINDOW_SIZE x input_size ]
        Output:
            - [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size]
        '''
        K = torch.einsum('bwi,io->bwo', inputs_for_keys, self.Wk)
        V = torch.einsum('bwi,io->bwo', inputs_for_values, self.Wv)
        Q = torch.einsum('bwi,io->bwo', inputs_for_queries, self.Wq)

        weights = self.atten_matrix([K,Q])
        return torch.einsum('bqk,bko->bqo', weights, V)

class MultiHeadedAttention(nn.Module):
    def __init__(self, emb_sz, use_mask, num_heads, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)
        self.use_mask = use_mask
        self.num_heads = 3
        self.embed_size = emb_sz
        # **The remainder in features are truncated off**
        self.head_size = self.embed_size // self.num_heads
        self.atten_heads = nn.ModuleList([AttentionHead(self.head_size, self.head_size, self.use_mask) for i in range(self.num_heads)])
        # Project the concatenated features back to embed_size
        self.linear_layer = nn.Linear(self.head_size * self.num_heads, self.embed_size)
        
    def forward(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        # Compute Heads
        Heads = torch.cat([head(inputs_for_keys[:, :, i*self.head_size : (i+1)*self.head_size],\
                               inputs_for_values[:, :, i*self.head_size : (i+1)*self.head_size],\
                               inputs_for_queries[:, :, i*self.head_size : (i+1)*self.head_size])\
                          for i,head in enumerate(self.atten_heads)], axis = -1)
        # Project back the concatenated features
        return self.linear_layer(Heads)

class TransformerEncoder(nn.Module):
    '''
    Transformer Encoder Block that reweights the input tensor based on self-attention
    '''
    def __init__(self, emb_sz, MultiHeaded = True, num_heads = 3, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        
        self.embed_size = emb_sz
        self.is_multi_headed = MultiHeaded
        self.num_heads = num_heads
        # Feed Forward Layer
        self.ff_layer = nn.Linear(self.embed_size, self.embed_size)

        self.self_atten = AttentionHead(self.embed_size, self.embed_size, True) if not self.is_multi_headed else MultiHeadedAttention(self.embed_size, True, self.num_heads)
        # Layer Normalization for stable gradients
        self.layer_norm = nn.LayerNorm(self.embed_size)
        
    def forward(self, inputs):
        '''
        1. Masked Self-Attention on the inputs
        2. Residual Connection and Layer Normalization
        3. Feed Forward Layer
        4. Residual Connection and Layer Normalization
        5. return leaky_relu of tensor

        - inputs: [n_sentences, max_n_words, embedding_size]
        - outputs: [n_sentences, max_n_words, embedding_size]
        '''
        # 1
        atten_inputs = self.self_atten(inputs, inputs, inputs)
        # 2
        atten_inputs = atten_inputs + inputs # Residual Part
        atten_inputs = self.layer_norm(atten_inputs)

        # 3
        outputs = self.ff_layer(atten_inputs)
        # 4
        outputs = outputs + atten_inputs # Residual Part
        outputs = self.layer_norm(outputs)
        return F.leaky_relu(outputs)


class TransformerBlocks(nn.Module):
    '''
    - Position Encoding
    - Transfomer Encoder **** Can use multiple tranformer encoders
    - MLP Classifier
    **** Requires number of Words **** 
    '''
    def __init__(self, emb_sz, num_transformer = 1, MultiHeaded = True, num_heads = 3, **kwargs):

        super(TransformerBlocks, self).__init__(**kwargs)
        #self.n_words = n_words
        self.embed_size = emb_sz

        # Pos Encoding
        self.pos_encoding = PositionalEncoding()

        # Transformer Encoder
        transformer_blocks = [TransformerEncoder(self.embed_size, MultiHeaded, num_heads) for _ in range(num_transformer)]
        self.encoder = nn.Sequential(*transformer_blocks)

        # Classifier (with sigmoid)
        #self.classifier = LinearClassifier(self.embed_size)

    def forward(self, inputs):
        inputs = self.pos_encoding(inputs)
        probs = self.encoder(inputs)
        return probs


'''
Positional Encoding for Input Seq in Transformer
'''
def positional_encoding(length, depth, device = 'cpu'):
    ## REFERENCE: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
    # Sinosoidal positional encoding
    half_depth = depth//2 
    #print('half_depth = ', half_depth)
    # Generate a range of positions and depths
    positions = torch.arange(length, device = device).unsqueeze(dim = 1) # (length, 1)
    #print('positions = ', positions)
    depths = torch.arange(half_depth, device = device).unsqueeze(dim = 0) / half_depth # (1, depth)
    #print('depths = ', depths)
    # Compute range of radians to take the sine and cosine of
    angle_rates = 1 / (10000**depths)    # (1, depth)
    #print('angle_rates = ', angle_rates)
    angle_rads = positions * angle_rates # (length, depth)
    #print('angle_rads = ', angle_rads)
    if depth % 2:
        pos_encoding = torch.cat([torch.sin(angle_rads), torch.zeros(len(angle_rads), 1, device = device), torch.cos(angle_rads)], axis = -1).float()
    else:
        pos_encoding = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], axis = -1).float()
    return pos_encoding

class PositionalEncoding(nn.Module):
    '''
    Add the Designed Sinosoidal Position-Encodings to the Inputs
    # outputs: [batch_sz, n_words, embedding_size]
    '''
    def __init__(self):
        super(PositionalEncoding, self).__init__()

        #self.n_words = n_words
        #self.embed_size = emb_sz
        self.pos_encoding = positional_encoding #(self.n_words, self.embed_size)

    def forward(self, x):
        # Scale
        embed_size = x.shape[-1]
        n_words = x.shape[-2]
        x *= torch.math.sqrt(float(embed_size))
        # Pos Encoding
        x += self.pos_encoding(n_words, embed_size, x.device).unsqueeze(dim = 0)
        return x
