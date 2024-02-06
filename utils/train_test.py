import torch

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import confusion_matrix

from metrics.metrics import f1_by_cm, masked_bce_loss, relevance_accuracy, total_loss
'''
Train and Test Section
'''
## Training Subroutines ##

def train_step(model, 
               train_data, 
               metrics, optimizer):
    '''
    The Generic Train Step for a Trainable Model

    Input:
        - model     : 
        - train_data:
            - word_embeddings         : Input data [n_sentences, n_words, embedding_dim]
            - eeg_features            : Input data [num_sentences, max_num_words, eeg_feature_size = 5460]
            - gaze_features           : Input data [n_sentences, n_words, feature_dim = 12]
            - true_labels             : Ground Truth [n_sentences, n_words] 
            - sen_lengths             : Sentence Lengths [n_sentences]
        - metrics   : [loss_func, acc_func]
        - optimizer :

    Output:
        - loss      :
        - acc       :
    '''
    model.train()
    # Unpacking
    word_embeddings, eeg_features, gaze_features, true_labels, sen_lengths = train_data
    loss_func, acc_func = metrics

    # Forward Pass
    predictions = model(word_embeddings, eeg_features, gaze_features)

    # Loss calculation
    loss = loss_func(predictions, true_labels, sen_lengths)
    
    # Backward Pass
    optimizer.zero_grad() # clear the gradient
    loss.backward()
    optimizer.step()

    # Calculate the acc ######## acc after the backward pass
    with torch.no_grad():
        model.eval()
        updated_preds = model(word_embeddings, eeg_features, gaze_features)
        acc = acc_func(updated_preds, true_labels, sen_lengths)
        model.train()
    return loss.clone().detach(), acc

def shuffle_embeddings(train_data):
    '''
    Shuffle the inputs and the labels
    Input:
        - word_embeddings : [n_sentences, n_words, embedding_dim]
        - eeg_features    : [num_sentences, max_num_words, eeg_feature_size = 5460]
        - gaze_features   : [n_sentences, n_words, feature_dim = 12]
        - true_labels     : [n_sentences, n_words] 
        - sen_lengths     : [n_sentences]
    '''
    #indices = torch.randperm(true_labels.shape[0])
    #return word_embeddings[indices], gaze_features[indices], true_labels[indices]
    indices = torch.randperm(train_data[-2].shape[0])
    #print('shuffled indices length:', len(indices))
    return [data[indices] for data in train_data]


def train_model(model, batch_size, epochs,
                train_data, test_data, downsample = True,
                metrics = [total_loss, relevance_accuracy],
                opt_type = 'SGD',
                lr = 0.05, min_lr = -1):
    '''
    The Overall Train Method for the Model

    Input:
        - model           :
        - batch_size      : number of sentences used for training
        - epochs          : number of the total epochs
        - train_data      :
            - word_embeddings : [n_sentences, n_words, embedding_dim]
            - eeg_features    : [num_sentences, max_num_words, eeg_feature_size = 5460]
            - gaze_features   : [n_sentences, n_words, feature_dim = 12]
            - true_labels: [n_sentences, n_words] 
            - sen_lengths: [n_sentences]

        - test_data       :
        - downsample      : if downsampling the train data
        - metrics         : [loss_func, acc_func]
        - opt_type        : SGD or Adam (maybe second order optimizer e.g. L-BFGS )
        - lr              : learning rate 
        - min_lr          : minimum lr for learning-rate scheduler; do not use scheduler if min_lr = -1

    Output:
        - loss_list       : batch losses
        - acc_list        :
    '''
    loss_list = []
    acc_list = []

    #word_embeddings, eeg_features, gaze_features, true_labels, sen_lengths = train_data

    # Define optimizers *** Adam By Default
    if 'SGD' == opt_type:
        opt = optim.SGD(model.parameters(), lr = lr)
    elif 'Adam' == opt_type:
        opt = optim.Adam(model.parameters(), lr = lr)
    else:
        raise ValueError(f'opt_type = {opt_type} not yet supported.')
        
    # (Optional) Learning Rate Scheduler 
    if -1 != min_lr:
        scheduler = CosineAnnealingLR(opt, T_max = epochs, eta_min = min_lr)

    n_sentences = len(train_data[-1])
    n_batches = n_sentences // batch_size
    # For each epoch
    for e in range(epochs + 1):
        # shuffle the whole input
        shuffled_data = shuffle_embeddings(train_data)
        if downsample:
            shuffled_data = down_sample(shuffled_data)
        # Batch the input 
        for startIdx in range(0, n_sentences, batch_size):
            #curr_WE = WE[startIdx: startIdx + batch_size]
            #curr_GF = GF[startIdx: startIdx + batch_size]
            #curr_TL = TL[startIdx: startIdx + batch_size]
            #curr_SL = SL[startIdx: startIdx + batch_size]
            curr_batch = [d[startIdx: startIdx + batch_size] for d in shuffled_data]
            loss, acc = train_step(model,
                                   curr_batch, 
                                   metrics, opt)
            # record the results
            loss_list.append(loss)
            acc_list.append(acc)

        if -1 != min_lr:
            scheduler.step()
        # Every 10 Epoch: Print out last batch loss and test loss
        if 0 == e % 10:
            test_loss, test_acc, cm, test_f1, (preds, labels, valid_indices) = test_step(model,
                                  test_data,
                                  metrics)
            
            print(f"\r[At Epoch{e}]: \t-- LastBatchLoss: {loss:.4f}, LastBatchAcc: {acc:.4f}, TestLoss: {test_loss:.4f}, TestAcc: {test_acc:.4f}, TestF1: {test_f1:.4f}", end = '')

    print()
    return (loss_list, acc_list), (test_loss, test_acc), cm, test_f1, (preds, labels, valid_indices)


## Testing Subroutines ##

def test_step(model,
              test_data,
              metrics = [total_loss, relevance_accuracy]):
    '''
    Generic Test Step for a Model
    '''
    loss_func, acc_func = metrics

    word_embeddings, eeg_features, gaze_features, true_labels, sen_lengths = test_data

    # Eval mode
    model.eval()

    with torch.no_grad():
        # Forward Pass
        predictions = model(word_embeddings, eeg_features, gaze_features)
    
        # Loss calculation
        loss = loss_func(predictions, true_labels, sen_lengths)
    
        # Acc ####
        acc = acc_func(predictions, true_labels, sen_lengths)

        # Confusion Matrix
        # Excluding padding
        #max_num_words = sen_lengths.max()
        max_num_words = true_labels.shape[-1]
        n_sen = len(sen_lengths)
        device = predictions.device

        valid_indices = torch.arange(max_num_words, device = device).expand(n_sen, max_num_words) < sen_lengths.unsqueeze(1)
        masked_preds = (predictions[valid_indices] >= 0.5).int().flatten()
        masked_labels = true_labels[valid_indices].flatten()

        cm = confusion_matrix(masked_labels.clone().detach().cpu().numpy(), masked_preds.clone().detach().cpu().numpy())
        # F1 Score
        test_f1 = f1_by_cm(cm)
        
    # Set back to train mode
    model.train()
    return loss.clone().detach(), acc, cm, test_f1, (predictions.clone().detach(), true_labels, valid_indices)
