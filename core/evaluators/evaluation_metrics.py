""" 
This file contains a collection of evaluation metrics with plots for the evaluator module.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef)


def create_dataframe(classes, ground_truths, vid_paths):
    """ Create a dataframe of classification results. """
    df = pd.DataFrame({
        "Predicted_Class": classes,
        "Ground_Truth": ground_truths,
        "Video_Path": vid_paths
    })
    return df


def create_classification_report(truth, pred, labels=["Class 0", "Class 1"]):
    """ Create a scikit-learn classification report as a matplotlib image. """
    report = classification_report(y_true=truth, y_pred=pred, target_names=labels, digits=3, zero_division=0, output_dict=True)

    # add support (number of classes) to the report
    class_report = {}
    for key, value in report.items():
        if isinstance(value, dict) and isinstance(value['support'], int):
            new_key = key + f" ({value['support']})"
            class_report[new_key] = value
        else:
            class_report[key] = value

    accuracy = accuracy_score(y_true=truth, y_pred=pred)
    ck_score = cohen_kappa_score(y1=truth, y2=pred)
    mc_coef = matthews_corrcoef(y_true=truth, y_pred=pred)

    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Classification Report', fontsize=20)
    plt.figtext(.25, .94, f"Accuracy: {'{:.2%}'.format(accuracy)}", fontsize=15)
    plt.figtext(.25, .92, f"Cohen’s Kappa score: {round(ck_score, 4)}", fontsize=15)
    plt.figtext(.25, .90, f"Matthew’s correlation coefficient: {round(mc_coef, 4)}", fontsize=15)

    df_cr = pd.DataFrame(class_report)
    plot = sn.heatmap(df_cr.iloc[:-1, :].T, linewidth=0, annot=True, fmt='.2%', cbar_kws={"shrink": 0.6}, annot_kws={'size': 12}).get_figure()
    plt.xticks(rotation=0, fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.25)
    return plot


def create_confusion_matrix(truth, pred, labels=["Class 0", "Class 1"]):
    """ Create a scikit-learn confusion matrix as a matplotlib image. """
    report = classification_report(y_true=truth, y_pred=pred, target_names=labels, digits=3, zero_division=0, output_dict=True)
    cf_matrix = confusion_matrix(y_true=truth, y_pred=pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix),
                         index=[i for i in labels],
                         columns=[i for i in labels])

    yticks = df_cm.index
    xticks = df_cm.columns
    annot = False if len(labels) > 10 else True

    accuracy = accuracy_score(y_true=truth, y_pred=pred)
    ck_score = cohen_kappa_score(y1=truth, y2=pred)
    mc_coef = matthews_corrcoef(y_true=truth, y_pred=pred)

    # Extract TP, TN, FP, FN from the confusion matrix
    TP = cf_matrix[1, 1]  # True Positive
    TN = cf_matrix[0, 0]  # True Negative
    FP = cf_matrix[0, 1]  # False Positive
    FN = cf_matrix[1, 0]  # False Negative

    # Calculate specific metrics
    precision = TP / (TP + FP) if TP + FP > 0 else 0  # Avoid division by zero
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0  # Avoid division by zero
    specificity = TN / (TN + FP) if TN + FP > 0 else 0  # Avoid division by zero

    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Confusion Matrix', fontsize=20)
    plt.figtext(.25, .90, f"Accuracy: {'{:.2%}'.format(accuracy)}", fontsize=15)
    plt.figtext(.25, .88, f"Precision: {'{:.2%}'.format(precision)}", fontsize=15)
    plt.figtext(.25, .86, f"Sensitivity: {'{:.2%}'.format(sensitivity)}", fontsize=15)
    plt.figtext(.25, .84, f"Specificity: {'{:.2%}'.format(specificity)}", fontsize=15)
    plt.figtext(.25, .82, f"Cohen’s Kappa score: {round(ck_score, 4)}", fontsize=15)
    plt.figtext(.25, .80, f"Matthew’s correlation coefficient: {round(mc_coef, 4)}", fontsize=15)

    plot = sn.heatmap(df_cm, linewidth=0, yticklabels=yticks, xticklabels=xticks, annot=annot, fmt='.2%', cbar_kws={"shrink": 0.6}, annot_kws={'size': 18},  square=True).get_figure()
    plt.xticks(rotation=0, fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Labels", fontsize=18, labelpad=40)
    plt.ylabel("True Labels", fontsize=18) 
    plt.subplots_adjust(left=0.25, top=0.85)
    return plot
