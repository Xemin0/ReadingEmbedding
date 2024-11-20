import torch

from data.data_loader import load_data
from model.reading_embedding_model import ReadingEmbeddingModel
from utils.train_test import train_model
from metrics.metrics import total_loss, relevance_accuracy
from utils.plotting import plot_tsne, plot_cm
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
feat_choice = [0, 1, 0]
epochs = 480
opt_type = 'SGD' # Adam

downsampled_data = load_data(downsample = True, subIdx = idx) #### Down Sample or not 
original_data = load_data(downsample = False, subIdx = idx)

embeddings, eeg_features, gaze_features, labels, sen_len = original_data

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
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# seed gpus if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
print(device)

# Initialize Model
mymodel = ReadingEmbeddingModel(word_embedding_dim=768,
                              eeg_feature_dim = 5460,
                              eye_gaze_feature_dim=12,
                              projection_dim=128,
                              n_trans = 1,
                              feat_choice = feat_choice,
                              MultiHeaded = True, num_heads = 3)

# Set Device
mymodel.to(device)
downsampled_data_device = [d.to(device) for d in downsampled_data]

d_embeddings, d_eeg_features, d_gaze_features, d_labels, d_sen_len = downsampled_data
# Train
(loss_list_real, acc_list_real), (test_loss, test_acc), cm, f1, (preds, labels, valid_indices) = train_model(mymodel, batch_size = 16, epochs = epochs,
            train_data = downsampled_data_device, test_data = downsampled_data_device, downsample = False,
            metrics = [total_loss, relevance_accuracy],
            opt_type = opt_type,
            lr = 0.05, min_lr = -1)


'''
Plotting ReadingEmbeddings with t-SNE
'''
# Get ReadingEmbeddings
reading_emb = mymodel.get_reading_embeddings(d_embeddings, d_eeg_features, d_gaze_features).clone().detach()

# Remove Paddings
re_no_pad = torch.cat([sen[:l] for sen, l in zip(reading_emb, d_sen_len)], dim = 0)
lb_no_pad = torch.cat([sen[:l] for sen, l in zip(d_labels, d_sen_len)], dim = 0)


if 2 == sum(f_choice):
    f_type = 'EEG + Eye_Gaze'
elif 1 == sum(f_choice):
    if f_choice[1]:
        f_type = 'EEG'
    elif f_choice[2]:
        f_type = 'EyeGaze'
    else:
        f_type = ' '
else:
    f_type = ' '

# Plot t-SNE
_ = plot_tsne(re_no_pad,
             lb_no_pad,
             'Reading Embeddings from ' + f_type)

# Plot Confusion Matrix
plot_cm(cm, 'Confusion Matrix on Training Data')
