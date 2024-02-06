import torch
import numpy as np


from .train_test import shuffle_embeddings, train_model
from metrics.metrics import masked_bce_loss, relevance_accuracy, f1_by_cm, total_loss
from data.data_loader import down_sample
from model.reading_embedding_model import ReadingEmbeddingModel

'''
K-Fold Cross Validation
'''
def k_fold_division(data, k_fold = 5):
    '''
    Shuffle and Divide the Data into K-Fold
    Input:
        - data:
            - word_embeddings
            - eeg
            - eye_gaze
            - labels
            - sen_len
    Output:
        - List of K [train_data, test_data] pairs of the original data
    '''
    # Shuffle 
    data = shuffle_embeddings(data)
    # Divide
    n_samples = len(data[-1])
    fold_size = n_samples // k_fold
    
    k_fold_datalist = []
    for k in range(k_fold - 1): # Process the first (k-1) fold 
        test_start = k*fold_size
        test_end = (k + 1) * fold_size

        test_data  = [d[test_start : test_end] for d in data]
        train_data = [torch.cat([d[ : test_start], d[test_end :]], dim = 0) for d in data]
        k_fold_datalist.append([train_data, test_data])

    # Process the remainder fold
    test_data = [d[-fold_size : ] for d in data]
    train_data = [d[ : -fold_size] for d in data]
    k_fold_datalist.append([train_data, test_data])
    return k_fold_datalist


def k_fold_CV(k_fold, 
              batch_size, epochs,
              data, downsample = True,
              n_trans = 1,
              feat_choice = [1, 1, 1], ## Word Embedding, EEG, EyeGaze
              metrics = [total_loss, relevance_accuracy], 
              opt_type = 'SGD',
              lr = 0.05, min_lr = -1,
              MultiHeaded = True, num_heads = 3, device = 'cpu'):
    '''
    Test on 1-K th proportion of data + Train on the rest
    Average across the K accuracies
    '''
    # Shuffle and Divide
    k_fold_datalist = k_fold_division(data, k_fold)

    assert k_fold == len(k_fold_datalist), f'The number K = {k_fold} does not match number of the divided datasets = {len(k_fold_datalist)}'
    # Apply the Train and Test Procedures
    total_test_acc = 0.0
    test_acc_list = []
    overall_cm = np.zeros((2,2)) # Assuming for Binary Classification
    overall_masked_preds = []
    overall_masked_labels = []
    for i, (train_d, test_d) in enumerate(k_fold_datalist):
        # Declaration of Model ### Get the constants from input
        REmodel = ReadingEmbeddingModel(word_embedding_dim=768, 
                                      eeg_feature_dim = 5460,
                                      eye_gaze_feature_dim=12, 
                                      projection_dim=128,
                                      n_trans = n_trans,
                                      feat_choice = feat_choice,
                                      MultiHeaded = MultiHeaded, num_heads = num_heads)
        print(f'At {i + 1}-th Fold:')
        if downsample: # Downsample the training and testing data for each model
            train_d = down_sample(train_d)
            test_d = down_sample(test_d)

        # Assign Device
        REmodel.to(device)
        train_d = [d.to(device) for d in train_d]
        test_d = [d.to(device) for d in test_d]
        
        _ , (test_loss, test_acc), fold_cm, fold_f1, (preds, labels, valid_indices)  = train_model(REmodel, batch_size = batch_size, epochs = epochs,
                                            train_data = train_d, test_data = test_d, downsample = False,
                                            metrics = metrics, 
                                            opt_type = opt_type,
                                            lr = lr, min_lr = min_lr)
        total_test_acc += test_acc
        test_acc_list.append(test_acc)
        overall_cm += fold_cm
        overall_masked_preds.append(preds[valid_indices])
        overall_masked_labels.append(labels[valid_indices])
    # Over All F1 Score ### or average?
    overall_f1 = f1_by_cm(overall_cm)
    overall_masked_preds = torch.cat(overall_masked_preds, dim = 0)
    overall_masked_labels = torch.cat(overall_masked_labels, dim = 0)
    # Average
    return total_test_acc / k_fold, test_acc_list, overall_cm, overall_f1, (overall_masked_preds, overall_masked_labels)
