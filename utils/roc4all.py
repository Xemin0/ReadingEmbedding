import torch

from data.data_loader import load_data
from utils.cross_validation import k_fold_CV
from metrics.metrics import total_loss, relevance_accuracy
from utils.plotting import plot_all_roc #, plot_tsne, plot_cm
'''
Load Real Data
- embeddings    : [num_sentences, max_num_words, hidden_size]
- eeg_features  : [num_sentences, max_num_words, eeg_feature_size = 5460]
- gaze_features : [num_sentences, max_num_words, num_measure_mode = 12]
- labels        : [num_sentences, max_num_words]
'''

def get_stats4ROC(feat_choice = [0, 1, 1], n_subs = 9,
                  epochs = 500, optimizer_type = 'Adam', n_translayer = 1, k = 5):
    # Loop Through each Subject:
    avg_acc_list = []
    cm_list = []
    f1_list = []
    pred_label_list = []

    # Get Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for idx in range(n_subs):
        # Load data
        original_data = load_data(downsample = False, subIdx = idx)
        #embeddings, eeg_features, gaze_features, labels, sen_len = original_data

        avg_acc, test_acc_list, overall_cm, overall_f1, pred_label = k_fold_CV(k_fold = k,
                                      batch_size = 16, epochs = epochs,
                                      data = original_data, downsample = True,
                                      n_trans = n_translayer, feat_choice = feat_choice,
                                      metrics = [total_loss, relevance_accuracy],
                                      opt_type = optimizer_type,
                                      lr = 0.05,
                                      MultiHeaded = True, num_heads = 3, device = device)
        avg_acc_list.append(avg_acc)
        cm_list.append(overall_cm)
        f1_list.append(overall_f1)
        pred_label_list.append(pred_label)

    return avg_acc_list, cm_list, f1_list, pred_label_list


# feat_choice = [WordEmbedding, EEG, EyeGaze]
f_choice = [0, 0, 1] # EyeGaze only
avg_acc_list, cm_list, f1_list, pred_label_list = get_stats4ROC(f_choice, n_subs = 9,
                                                                   epochs = 500, optimizer_type = 'Adam', n_translayer = 1, k = 5)

# Plot ROC 
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
plot_all_roc(pred_label_list, f_type = 'EyeGaze')
