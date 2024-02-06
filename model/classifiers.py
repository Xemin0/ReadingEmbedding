'''
Classifiers as Submodels for plug-and-play
'''
import torch
import torch.nn as nn
#import torch.nn.functional as F

class LinearClassifier(nn.Module):
    '''
    Simple Classifier for High- Low- Relevance Inference

    Input:
        - Reading Embeddings : [n_sentence, n_words, projection_dim]

    Output:
        - Labels             : [n_sentence, n_words]
    '''
    def __init__(self, projection_dim, **kwargs):
        super(LinearClassifier, self).__init__(**kwargs)
        self.linearLayer = nn.Sequential(
                                    nn.Linear(projection_dim, 1),
                                    nn.Sigmoid()
                                    )

    def forward(self, embeddings):
        return self.linearLayer(embeddings).squeeze(dim = -1)
