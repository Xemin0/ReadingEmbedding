import torch
import torch.nn.functional as F

'''
Metrics
** Consider Masked_BCE_Loss + F1_Score
'''
def relevance_accuracy(y_pred, y_true, sen_lengths):
    '''
    Accuracy between the predicted and the true label
    [n_sentences, n_words]

    ** the actual sentence length may differ **
    '''
    # Threshold the Probabilistic Predicitons to Labels
    binary_pred = y_pred >= 0.5
    y_true = y_true.type_as(binary_pred)

    device = y_true.device

    # Compare
    correctness = (binary_pred == y_true)

    # Mask out the padding according to the given sentence lengths
    max_len = y_true.shape[-1]
    mask = torch.arange(max_len, device = device).expand(len(sen_lengths), max_len) < sen_lengths.unsqueeze(-1)
    acc = correctness.masked_select(mask).float().mean()
    return acc.item()


def f1_by_cm(cm):
    TP = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TN = cm[1,1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

'''
Loss
'''
def masked_bce_loss(y_pred, y_true, sen_lengths):
    '''
    BCE Loss that with a Mask for Variable Sententce Lengths
    '''
    device = y_true.device
    max_len = y_true.shape[-1]
    mask = torch.arange(max_len, device = device).expand(len(sen_lengths), max_len) < sen_lengths.unsqueeze(-1)

    masked_preds = y_pred.masked_select(mask)
    masked_labels = y_true.masked_select(mask)

    # BCE for the masked inputs
    loss = F.binary_cross_entropy(masked_preds, masked_labels)

    return loss


def masked_mse_loss(y_pred, y_true, sen_lengths):
    '''
    MSE Loss with a Mask for Variable Sentence Lengths
    '''
    device = y_true.device
    max_len = y_true.shape[-1]
    mask = torch.arange(max_len, device = device).expand(len(sen_lengths), max_len) < sen_lengths.unsqueeze(-1)

    masked_preds = y_pred.masked_select(mask)
    masked_labels = y_true.masked_select(mask)

    squared_diff = (masked_preds - masked_labels) ** 2
    return squared_diff.mean(dim = 0)


def masked_f1_loss(y_pred, y_true, sen_lengths):
    '''
    Soft F1 Loss based on Bray-Curtis Distance
    '''
    device = y_true.device
    max_len = y_true.shape[-1]
    mask = torch.arange(max_len, device = device).expand(len(sen_lengths), max_len) < sen_lengths.unsqueeze(-1)

    masked_preds = y_pred.masked_select(mask).float()
    masked_labels = y_true.masked_select(mask).float()

    # Calculate the TP, TN, FP, FN
    tp = torch.sum(y_true * y_pred, dim = 0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), dim = 0)
    fp = torch.sum((1 - y_true) * y_pred, dim = 0)
    fn = torch.sum(y_true * (1 - y_pred), dim = 0)

    # Precesion and Recall
    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    # F1 Score
    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return 1 - torch.mean(f1)

def total_loss(y_pred, y_true, sen_lengths):
    '''
    Total Loss consisting of Masked-:
    - BCE Loss:
    - MSE Loss:
    - F1 Loss:
    '''
    m_bce = masked_bce_loss(y_pred, y_true, sen_lengths)
    m_mse = masked_mse_loss(y_pred, y_true, sen_lengths)
    m_f1 = masked_f1_loss(y_pred, y_true, sen_lengths)
    return m_bce + m_mse + 0.5 * m_f1
