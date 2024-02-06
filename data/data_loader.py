'''
Data Loader
Input:
    - sample_down   : if sampling down the binary labels to be of the same quantity
Output:
    - embeddings    : [num_sentences, max_num_words, hidden_size]
    - EEG           : [num_sentences, max_num_words, eeg_feature_size = 5460]
    - gaze_features : [num_sentences, max_num_words, num_measure_mode = 12]
    - labels        : [num_sentences, max_num_words]
    - sen_len       : [num_sentences]
'''
import torch
import sys
import json
import scipy.io as sio
import numpy as np

import random
import math


def load_data(downsample = False, subIdx = 0, rootpath = './Datasets/'):
    # List of Subject Names
    subject_name_list = ['ZAB', 'ZDM', 'ZDN', 'ZJM', 'ZJN', 'ZJS', 'ZKH', 'ZKW', 'ZMG']
    
    # Word Embeddings with padding
    em_data = sio.loadmat(rootpath + 'wordEmbeddingAll.mat')['wordEmbeddingAll'].squeeze() # (358, )
    
    sen_len_em = [m.shape[0] for m in em_data]
    max_num_words_em = max(sen_len_em)
    em_size = em_data[0].shape[1]
    embeddings = torch.zeros((len(em_data), max_num_words_em, em_size), dtype = torch.float32)
    
    for i, m in enumerate(em_data):
        n_words = m.shape[0]
        # Add Positional Encoding ???
        embeddings[i, : n_words] = torch.from_numpy(m)

    ########################################
    # EEG Feature Data with padding
    EEG_path = rootpath + 'EEG_data/EEGEmbeddingAll_' + subject_name_list[subIdx] + '.mat'
    eeg_data = sio.loadmat(EEG_path)['EEGEmbeddingAll'].squeeze() # (358,)

    sen_len_eeg = [len(m) for m in eeg_data]
    max_num_words_eeg = max(sen_len_eeg)
    for i, sen in enumerate(eeg_data):
        if sen[0][0].shape[0] > 1:
            eeg_feat_size = sen[0][0].shape[0]
    
    eeg_features = torch.zeros((len(eeg_data), max_num_words_eeg, eeg_feat_size), dtype = torch.float32)

    for i, sen in enumerate(eeg_data):
        for j, word in enumerate(sen):
            if word[0].shape[0] > 1:
                eeg_features[i, j, :] = torch.from_numpy(word[0].squeeze())
    ########################################    
    # Eye Gaze Data with padding
    EGD_path = rootpath + 'subject_data/eyeGazeData_' + subject_name_list[subIdx] + '.json'
    with open(EGD_path) as file:
        eg_data = json.load(file)

    sen_len_eg = [len(d) for d in eg_data]
    max_num_words_eg = max(sen_len_eg)
    gaze_features = torch.zeros((len(eg_data), max_num_words_eg, 12), dtype = torch.float32)

    # Extract data from the json data
    for i, sentence in enumerate(eg_data):
      for j, word in enumerate(sentence): ## Sentence length / num_word per sentence may vary
        gaze_features[i, j, 0] = word['nFixations']
        gaze_features[i, j, 1] = word['meanPupilSize']
        gaze_features[i, j, 2] = word['FFD']
        gaze_features[i, j, 3] = word['FFD_pupilsize']
        gaze_features[i, j, 4] = word['TRT']
        gaze_features[i, j, 5] = word['TRT_pupilsize']
        gaze_features[i, j, 6] = word['GD']
        gaze_features[i, j, 7] = word['GD_pupilsize']
        gaze_features[i, j, 8] = word['GPT']
        gaze_features[i, j, 9] = word['GPT_pupilsize']
        gaze_features[i, j, 10] = word['SFD']
        gaze_features[i, j, 11] = word['SFD_pupilsize']
          
    ########################################   
    # Labels with padding
    with open(rootpath + 'logicVectors.json') as file:
        lb_data = json.load(file)

    sen_len_lb = [len(d) for d in lb_data]
    max_num_words_lb = max(sen_len_lb)
    labels = torch.zeros((len(lb_data), max_num_words_lb), dtype = torch.float32)
    
    for i, sentence in enumerate(lb_data):
        n_words = len(sentence)
        labels[i, :n_words] = torch.tensor(sentence)

    ########################################   
    if downsample:
        return down_sample([embeddings, eeg_features, gaze_features, labels, torch.tensor(sen_len_em)])
        '''
        # Convert the padded tensors back to list of torch.tensors (without padding)
        em_list = [d[:l] for d, l in zip(embeddings, sen_len_em)] # Word Embedding
        eg_list = [d[:l] for d, l in zip(gaze_features, sen_len_eg)] # Eye Gaze Data
        lb_list = [d[:l] for d, l in zip(labels, sen_len_lb)] # Labels

        down_sampled_em = []
        down_sampled_eg = []
        down_sampled_lb = []
        
        # Sampling Down the Binary Labels to be of the same quantity
        # Count the total number of 0s and 1s in the labels
        n_ones = sum(sum(sentence) for sentence in lb_data)
        n_zeros = sum(l - sum(sentence) for sentence, l in zip(lb_data, sen_len_lb))
        # The down sampling rate for 0s
        rate = n_ones / n_zeros

        # DownSample each tensor in the list by seletively removing 0s
        for i, sentence in enumerate(lb_list):
            # indices of zeros
            zero_indices = torch.where(0 == sentence)[0]
            # randomly select a subset of zero indices to keep
            zeros2keep = int(len(zero_indices) * rate)
            kept_zero_indices = torch.tensor(
                                    np.random.choice(zero_indices, zeros2keep, replace = False)
                                )

            # Mask for elements to keep
            mask = torch.full(sentence.shape, False, dtype = bool)
            mask[1 == sentence] = True
            mask[kept_zero_indices] = True

            # Append the downsampled data
            down_sampled_em.append(em_list[i][mask])
            down_sampled_eg.append(eg_list[i][mask])
            down_sampled_lb.append(lb_list[i][mask])

        # Padding to convert lists back to tensors
        # Word Embedding
        sen_len_em = [m.shape[0] for m in down_sampled_em]
        max_num_words_em = max(sen_len_em)
        em_size = down_sampled_em[0].shape[1]
        embeddings = torch.zeros((len(down_sampled_em), max_num_words_em, em_size), dtype = torch.float32)
        
        for i, m in enumerate(down_sampled_em):
            n_words = m.shape[0]
            embeddings[i, : n_words] = m
            
        # Eye Gaze Data 
        sen_len_eg = [len(d) for d in down_sampled_eg]
        max_num_words_eg = max(sen_len_eg)
        gaze_features = torch.zeros((len(down_sampled_eg), max_num_words_eg, 12), dtype = torch.float32)

        for i, m in enumerate(down_sampled_eg):
            n_words = m.shape[0]
            gaze_features[i, :n_words] = m
            
        # Label
        sen_len_lb = [len(d) for d in down_sampled_lb]
        max_num_words_lb = max(sen_len_lb)
        labels = torch.zeros((len(down_sampled_lb), max_num_words_lb), dtype = torch.float32)

        for i, lb in enumerate(down_sampled_lb):
            n_words = len(lb)
            labels[i, :n_words] = lb
        '''
        
    return [embeddings, eeg_features, gaze_features, labels, torch.tensor(sen_len_em)]


##########

def down_sample(data):
    '''
    Down Sample tensors
    [embeddings, gaze_features, labels, torch.tensor(sen_len_em)]
    1. Remove Padding in each tensor
    2. DownSample the labels
    3. Create Mask
    4. DownSample other tensors using the Mask
    5. Pad
    '''
    sen_len = data[-1]
    labels = data[-2]
    # 1. Remove Paddings: list of tensors for each original tensor
    data_lists = [[d[:l] for d, l in zip(tensor, sen_len)] for tensor in data[:-1]]

    downsampled_lists = [[] for _ in range(len(data[:-1]))]
    
    # 2. Down Sample the Labels
    n_ones = sum(sum(sentence) for sentence in labels)
    n_zeros = sum(l - sum(sentence) for sentence, l in zip(labels, sen_len))
    # The down sampling rate for 0s
    rate = n_ones / n_zeros

    # 3. Creating Mask for each sentence
    for i, sentence in enumerate(data_lists[-1]): # labels
        # indices of zeros
        zero_indices = torch.where(0 == sentence)[0]
        # randomly select a subset of zero indices to keep
        if 0 == random.randint(0,1):
            zeros2keep = math.floor(len(zero_indices) * rate)
        else:
            zeros2keep = math.ceil(len(zero_indices) * rate)
        kept_zero_indices = torch.tensor(
                                np.random.choice(zero_indices, zeros2keep, replace = False)
                            )

        # Mask for elements to keep
        mask = torch.full(sentence.shape, False, dtype = bool)
        mask[1 == sentence] = True
        mask[kept_zero_indices] = True

        # 4. Append the downsampled data
        for new_list, old_list in zip(downsampled_lists, data_lists):
            new_list.append(old_list[i][mask])

    # 5. Padding to convert lists back to tensors
    padded_downsampled_data = []
    
    # Embeddings and Features 
    for new_list in downsampled_lists[:-1]:
        new_sen_len = [len(d) for d in new_list]
        new_max_num_words = max(new_sen_len)
        embed_size = new_list[0].shape[1]

        padded_tensor = torch.zeros((len(new_list), new_max_num_words, embed_size), dtype = torch.float32)
        for i, m in enumerate(new_list):
            n_words = m.shape[0]
            padded_tensor[i, :n_words] = m

        padded_downsampled_data.append(padded_tensor)

    # Label
    padded_labels = torch.zeros((len(downsampled_lists[-1]), new_max_num_words), dtype = torch.float32)

    for i, lb in enumerate(downsampled_lists[-1]):
        n_words = len(lb)
        padded_labels[i, :n_words] = lb

    padded_downsampled_data.append(padded_labels)
    padded_downsampled_data.append(torch.tensor(new_sen_len))
    return padded_downsampled_data
