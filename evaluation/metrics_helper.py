'''Help utilities to compute metrics for the CLAP model.'''
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, accuracy_score
import torch
import pandas as pd
import json


def compute_metrics(ground_truth, scores, normalize_scores=True):
    ''' Compute precision-recall pairs for different thresholds
    to compute the average precision.
    
    Args:
    - ground_truth: Ground truth labels.
    - scores: Predicted scores.
    - normalize_scores: If True, normalize the scores to [0, 1].
    
    Returns:
    - precision_dict: Precision for each class.
    - recall_dict: Recall for each class.
    - average_precision_dict: Average precision for each class.
    '''

    if normalize_scores:
        scores = (scores + 1) / 2

    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()

    for i in range(ground_truth.shape[-1]):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            ground_truth[:, i].flatten(),
            scores[:, i].flatten())
        
        average_precision_dict[i] = average_precision_score(
            ground_truth[:, i].flatten(),
            scores[:, i].flatten())

    # Compute micro-average ROC curve and ROC area
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(
        ground_truth.flatten(),
        scores.flatten())

    # average_precision_dict["micro"] = average_precision_score(
    #     ground_truth.flatten(),
    #     scores.flatten())

    return precision_dict, recall_dict, average_precision_dict


def compute_class_wise_accuracy(ground_truth, scores):

    # GT and scores comprise the indices of the active class
    # E.g.,  ground_truth = [0, 50, 100, 40], scores = [0, 50, 100, 40]

    # Compute the accuracy for each class
    
    accuracy_dict = dict()

   # Max N of classes
    n_classes = max(ground_truth) + 1

    for i in range(n_classes):
        # Get the indices of the examples for the class i
        indices = [j for j in range(len(ground_truth)) if ground_truth[j] == i]
        # Get the scores for the class i
        class_scores = scores[indices]
        # Get the ground truth for the class i
        class_gt = ground_truth[indices]
        # Compute the accuracy for the class i
        accuracy_dict[i] = accuracy_score(class_gt, class_scores)

    # Compute the mean accuracy
    # print("Mean accuracy: ", sum(accuracy_dict.values()) / len(accuracy_dict))

    return accuracy_dict



def compute_f1_score(ground_truth, scores):
    ''' Compute the F1 score for the given threshold.

    Args:
    - ground_truth: Ground truth labels.
    - scores: Predicted scores.

    Returns:
    - f1_score_dict: F1 score for each class.
    '''

    # use scikit-learn to compute the F1 score


    # Compute micro average F1 score
    micro_f1 = f1_score(ground_truth.flatten(), scores.flatten(), average='micro')

    # Compute macro average F1 score
    macro_f1 = f1_score(ground_truth.flatten(), scores.flatten(), average='macro')




    return micro_f1, macro_f1
