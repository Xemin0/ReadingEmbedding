import torch
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from scipy.stats import sem
from sklearn.utils import resample

import matplotlib.pyplot as plt
#from matplotlib.cm import get_cmap
import seaborn as sns

def plot_cm(cm, title = 'Overall Confusion Matrix in Cross Validation'):
    # Plotting
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot = True, ax = ax, fmt = 'g', cmap = 'Blues')
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['Low', 'High'])
    ax.yaxis.set_ticklabels(['Low', 'High'])
    
    plt.show()


def plot_roc_auc(y_true, y_preds):
    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    # AUC
    roc_auc = auc(fpr, tpr)
    # Plot
    plt.figure()
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characterisitc')
    plt.legend(loc = 'lower right')
    plt.show()


def plot_all_roc(p_l_list, f_type = 'EEG'):
    cmap = plt.colormaps['tab10']
    colors = cmap.colors
    #setup figure
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color = colors)
    # Calculate and plot each subject's roc
    for i, (preds, lbs) in enumerate(p_l_list):
        fpr, tpr, thresholds = roc_curve(lbs, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw = 2, label = f'subject[{i}]-{roc_auc:.2f}')

    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f_type)
    plt.legend(loc = 'lower right')
    plt.savefig('./Results/all_roc.png')


def plot_mean_roc_range(p_l_list_list, f_type_list = ['EEG']):
    cmap = plt.colormaps['tab10']
    colors = cmap.colors
    # setup figure
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color = colors)
    for p_l_list, f_type in zip(p_l_list_list, f_type_list):
        tprs = []
        base_fpr = np.linspace(0, 1, 101)  # A Common Scale for all FPRs

        # Calculate ROC for each subject
        for preds, lbs in p_l_list:
            fpr, tpr, _ = roc_curve(lbs, preds)
            roc_auc = auc(fpr, tpr)
            # Interpolate each TPR at the common FPR scale
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis = 0)
        std_tprs = tprs.std(axis = 0)

        # Upper and Lower Bounds
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = mean_tprs - std_tprs

        # Plotting 
        ax.plot(base_fpr, mean_tprs, lw = 1.5, label = f_type)
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, alpha = 0.2)

    ax.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves of Different Biomarkers')
    ax.legend(loc = 'lower right')
    plt.savefig('./Results/meanROCall.svg', dpi = 300, bbox_inches = 'tight')
    plt.show()


########

def plot_roc_auc_with_CI(y_true, y_scores, n_bootstraps = 1000, alpha = 0.95):
    # Calculate the ROC curve and AUC for the original data
    base_fpr, base_tpr, _ = roc_curve(y_true, y_scores)
    base_auc = auc(base_fpr, base_tpr)

    # Bootstrapping for confidence intervals
    tprs = []
    aucs = []
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = resample(np.arange(len(y_true)), replace = True)
        if len(np.unique(y_true[indices])) < 2:
            # If sample doesn't have both classes, skip
            continue

        # Calculate ROC for the resampled data
        fpr, tpr, _ = roc_curve(y_true[indices], y_scores[indices])
        tprs.append(np.interp(base_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(base_fpr, tprs[-1]))

    # Convert to numpy array for calculations
    tprs = np.array(tprs)
    aucs = np.array(aucs)

    # Mean and Std
    mean_tprs = tprs.mean(axis = 0)
    std = tprs.std(axis = 0)

    # Calculate confidence intervals
    tprs_upper = np.minimum(mean_tprs + std * sem(tprs), 1)
    tprs_lower = mean_tprs - std * sem(tprs)

    # Plotting
    plt.figure(figsize = (10, 8))
    plt.plot(base_fpr, base_tpr, color = 'darkorange', lw = 0.6, label = f'AUC = {base_auc:.2f}')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'darkorange', alpha = 0.3,
                    label = f'{alpha*100:.0f}% CI')
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC with Confidence Interval')
    plt.legend(loc = 'lower right')
    plt.show()


def plot_all_roc_with_CI(p_l_list, f_type = 'EEG', n_bootstraps = 1000, alpha = 0.95):
    cmap = plt.colormaps['tab10']
    colors = cmap.colors
    #setup figure
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color = colors)
    # Calculate and plot each subject's roc with CI
    for i, (preds, lbs) in enumerate(p_l_list):
        base_fpr, base_tpr, _ = roc_curve(lbs, preds)
        base_auc = auc(base_fpr, base_tpr)

        # Bootstrapping for confidence intervals
        tprs = []
        aucs = []
        for _ in range(n_bootstraps):
            # Resample with replace ment
            indices = resample(np.arange(len(lbs)), replace = True)
            if len(np.unique(lbs[indices])) < 2:
                # if sample doesn't have both classes skip
                continue

            # Calculate ROC for the sampled data
            fpr, tpr, _ = roc_curve(lbs[indices], preds[indices])
            tprs.append(np.interp(base_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(auc(base_fpr, tprs[-1]))

        # Convert to numpy array for calculations
        tprs = np.array(tprs)
        aucs = np.array(aucs)

        # Mean and std
        mean_tprs = tprs.mean(axis = 0)
        std = tprs.std(axis = 0)
        std /= np.sqrt(tprs.shape[0])

        # Calculate confidence intervals
        tprs_upper = np.minimum(mean_tprs + std * 1.96, 1)
        tprs_lower = mean_tprs - std * 1.96

        # Plotting
        plt.plot(base_fpr, base_tpr, lw = 0.6, label = f'subject[{i}]: AUC={base_auc:.2f}')
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, alpha = 0.3)

    ####
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f_type)
    plt.legend(loc = 'lower right')
    #plt.savefig('./Results/all_roc.png')



def plot_tsne(data, labels, relative_alpha = 0.7, title_str = ''):
    '''
    Plot t-SNE of data with corresponding labels
    Input: 
        - data : [n_samples, feature_dim] flattened
        - label: [n_samples] flattened
    '''
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # Apply t-SNE
    tsne = TSNE(n_components = 2, random_state = 42)
    tsne_results = tsne.fit_transform(data)

    # Plot with label-based coloring
    plt.figure(figsize = (5,5))
    for lb in np.unique(labels):
        indices = labels == lb
        if 0 == lb:
            label_str = 'Low Relevance'
            alpha_val = 1.0
        elif 1 == lb:
            label_str = 'High Relevance'
            alpha_val = relative_alpha
        else: 
            label_str = f'Label {lb}' # Generic Label for other classes
            alpha_val = 0.7
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label = label_str, alpha = alpha_val)

    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('t-SNE Visualization of High Dimensional ' + title_str)
    plt.legend()
    plt.savefig('./Results/' + title_str + '.svg', dpi = 300, bbox_inches = 'tight')
    plt.show()
    return tsne_results
