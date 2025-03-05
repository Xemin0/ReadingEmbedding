import torch

# For RandSeed
import numpy as np
import random

from data.data_loader import load_data
from model.reading_embedding_model import ReadingEmbeddingModel
from utils.cross_validation import k_fold_CV
from metrics.metrics import total_loss, relevance_accuracy
from utils.plotting import plot_tsne, plot_cm, plot_roc_auc
'''
Train the Model On Real Data
'''

'''
Load Real Data
- embeddings    : [num_sentences, max_num_words, hidden_size]
- eeg_features  : [num_sentences, max_num_words, eeg_feature_size = 5460]
- gaze_features : [num_sentences, max_num_words, num_measure_mode = 12]
- labels        : [num_sentences, max_num_words]
'''

### Load the data of of i-th subject
idx = 0 #### 0 - 8 for 9 different subjects
feat_choice = [0, 1, 0] # An indicator list/vector representing the choice of  [word_embedding, eeg_feats, eye_gaze_feats]
epochs = 500
n_translayer = 1
opt_type = 'SGD' # or 'Adam'
min_lr = -1 # not to use lr scheduler when -1

downsampled_data = load_data(downsample = True, subIdx = idx) #### Down Sample or not 
original_data = load_data(downsample = False, subIdx = idx)

embeddings, eeg_features, gaze_features, labels, sen_len = original_data
#d_embeddings, d_eeg_features, d_gaze_features, d_labels, d_sen_len = downsampled_data

# total number of 1s after downsample
#print('Total # of 1s after downsample = ', downsampled_data[-2].sum())

# total number of 0s before downsample
#print('Total # of 0s before downsample = ', sum(l - sum(s) for s, l in zip(original_data[-2], original_data[-1])))

# total number of 0s after downsample
#print('Total # of 0s after downsample = ', sum(l - sum(s) for s, l in zip(original_data[-2], original_data[-1])))

'''
# Device setup
'''
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# seed gpus if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
print(device)


avg_acc, test_acc_list, overall_cm, overall_f1, (overall_preds, overall_labels) = k_fold_CV(k_fold = 5,
                              batch_size = 40, epochs = epochs,
                              data = original_data, downsample = True,
                              n_trans = n_translayer, feat_choice = feat_choice,
                              metrics = [total_loss, relevance_accuracy],
                              opt_type = opt_type,
                              lr = 0.05, min_lr = min_lr,
                              MultiHeaded = True, num_heads = 3, device = device)



print('Average Acc = ', avg_acc)
print('Overall F1 Score = ', overall_f1)


# Plot ROC-AUC
plot_roc_auc(overall_labels, overall_preds)

# Plot Confusion Matrix
plot_cm(overall_cm, 'Confusion Matrix on Training Data')
