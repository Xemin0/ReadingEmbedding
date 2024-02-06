import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifiers import LinearClassifier
from .transformer import TransformerBlocks
'''
The Complete Pipeline of ReadingEmbedding Model
that administrates:
    - Generation of ReadingEmbeddings for words in each sentence
    - Classification of High-Low Relevances for words in each sentence
'''
class ReadingEmbeddingModel(nn.Module):
    '''
    One Reading-Embedding Encoder with a Classifier

    Input:
        - Word Embeddings    : [n_sentences, n_words, embedding_dim] ； ## !!! May need L2-Normalization 
        - EEG                : [num_sentences, max_num_words, eeg_feature_size = 5460] ## L2-Normalization
        - Eye Gaze Features  : [n_sentences, n_words, feature_dim = 12] ## L1-Normalization
    Intermediate Output:
        - Reading Embeddings : [n_sentences, n_words, projection_dim]
    Final Output:
        - Labels             : High- / Low- Relevance [(n_subjects), n_sentences, n_words]
    
    #######################################################################
    Method : 
        - Normalization
        - Project the input to a common space then element-wise addition (reading embeddings)
        - classifier (labels)
    '''
    def __init__(self, word_embedding_dim, eeg_feature_dim, eye_gaze_feature_dim, projection_dim, 
                 n_trans = 1, feat_choice = [1, 1, 1],
                 MultiHeaded = True, num_heads = 3):
        super(ReadingEmbeddingModel, self).__init__()

        self.embed_size = word_embedding_dim
        self.eeg_feature_dim = eeg_feature_dim
        self.eye_gaze_feature_dim = eye_gaze_feature_dim
        self.projection_dim = projection_dim
        
        # Linear layers for projecting embeddings
        self.feat_choice = feat_choice
        we, eeg, eg = feat_choice
        self.word_projection = nn.Linear(word_embedding_dim, projection_dim) if we else lambda x: 0 
        self.eeg_projection = nn.Linear(eeg_feature_dim, projection_dim) if eeg else lambda x: 0
        self.eye_gaze_projection = nn.Linear(eye_gaze_feature_dim, projection_dim) if eg else lambda x: 0

        # Transformer Blocks
        self.encoder = TransformerBlocks(projection_dim, n_trans, MultiHeaded, num_heads)
        # Classifier
        self.classifier = LinearClassifier(self.projection_dim)
            
    #---------------#

    def normalize_l1(self, vecs):
        '''
        Normalize by L1 Norm ******
        (Sum of all Entries in the Second last dimension)

        - vecs : [n_sentences, n_words, feature_dim = 12]
        '''
        magnitudes = vecs.sum(axis = -2, keepdim = True)
        # avoid 0 magnitudes for each feature_dim across words within each sentence
        magnitudes[0 == magnitudes] = 1.0
        return vecs / magnitudes
    
    def normalize_l2(self, vecs):
        '''
        Normalize by L2 Norm
        (Square Root of the Squared sum of all Entries in the last dimension)

        - vecs : [n_sentences, n_words, embedding_dim]
        '''
        magnitudes = torch.sqrt((vecs**2).sum(axis = -1, keepdim = True))
        # avoid 0 magnitudes for each word in each sentence
        magnitudes[0 == magnitudes] = 1.0
        return vecs / magnitudes

    #---------------#
    def get_reading_embeddings(self, word_embeddings, eeg_features, eye_gaze_features):
        # L2 - Normalize word_embeddings
        word_embeddings = self.normalize_l2(word_embeddings)

        # L2 - Normalize EEG features
        eeg_features = self.normalize_l2(eeg_features)

        # L1 - Normalize eye gaze features
        # Softmax across words for each sentence
        eye_gaze_features = self.normalize_l1(eye_gaze_features)
        eye_gaze_features = F.softmax(eye_gaze_features, dim = -2)

        # Project the inputs to a common space
        word_embeddings = self.word_projection(word_embeddings)
        eeg_features = self.eeg_projection(eeg_features)
        eye_gaze_features = self.eye_gaze_projection(eye_gaze_features)
        # Return the element-wise addition as the Reading Embeddings
        return self.encoder(word_embeddings + eeg_features + eye_gaze_features) # [n_sentences, n_words, projection_dim] 

    def forward(self, word_embeddings, eeg_features, eye_gaze_features):
        reading_embeddings = self.get_reading_embeddings(word_embeddings, eeg_features, eye_gaze_features)
        # Return the classifier's results
        return self.classifier(reading_embeddings) # [n_sentences, n_words] 

