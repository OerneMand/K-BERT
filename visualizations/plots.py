import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import auc, roc_curve 
from distutils.spawn import find_executable
from sklearn.preprocessing import label_binarize

plt.style.use('seaborn')
if find_executable('latex'):
    plt.style.use('tex')

predictions_directory = "outputs/job.probase_agnews/"
val_or_test = "val"

def roc_curve_plot(predictions_directory = predictions_directory, val_or_test = val_or_test):
    # Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    data = pd.read_csv(predictions_directory + f"{val_or_test}_predictions.csv")
    n_classes = sum([1 for name in list(data) if "class" in name])
    y_pred = data.iloc[:, :n_classes].values
    y_real = label_binarize(data.label, range(4))  

    lw = 2
    # Compute ROC curve and ROC area for each class 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes): 
        fpr[i], tpr[i], _ = roc_curve(y_real[:, i], y_pred[:, i]) 
        roc_auc[i] = auc(fpr[i], tpr[i]) 

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates 
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points 
    mean_tpr = np.zeros_like(all_fpr) 
    for i in range(n_classes): 
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i]) 
    
    # Finally average it and compute AUC 
    mean_tpr /= n_classes 
    
    fpr["macro"] = all_fpr 
    tpr["macro"] = mean_tpr 
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"]) 
    
    # Plot all ROC curves 
    plt.figure() 
    plt.plot(fpr["micro"], tpr["micro"], 
            label='micro-average ROC curve (area = {0:0.2f})' 
                ''.format(roc_auc["micro"]), 
            color='deeppink', linestyle=':', linewidth=lw*2) 
    
    plt.plot(fpr["macro"], tpr["macro"], 
            label='macro-average ROC curve (area = {0:0.2f})' 
                ''.format(roc_auc["macro"]), 
            color='navy', linestyle=':', linewidth=lw*2) 
    
    colors = cycle(['aqua', 'goldenrod', 'cornflowerblue', 'green']) 
    class_mapping = ["World", "Sports", "Business", "Sci/Tech"]
    for i, color in zip(range(n_classes), colors): 
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, 
                label=f"ROC curve of class {class_mapping[i]} (area = {roc_auc[i]:0.2f})") 
        #plt.fill_between(fpr[i], tpr[i], alpha = .2)
    
    plt.plot([0, 1], [0, 1], "k--", lw=lw*.5) 
    plt.xlim([0, 1]) 
    plt.ylim([0, 1]) 
    plt.xlabel("False Positive Rate") 
    plt.ylabel("True Positive Rate") 
    plt.legend(loc="lower right") 
    plt.savefig(f"visualizations/roc_{predictions_directory[12:-1]}_{val_or_test}.pdf", format = "pdf")

if __name__ == "__main__":
    roc_curve_plot()