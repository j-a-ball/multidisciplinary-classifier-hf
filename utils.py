import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def onehot(row):
    return [float(l.strip('[]')) for l in row['labelsI'].split()]


def fetch_data(train_, test_):

    # foscdata/train_train.csv 
    # foscdata/train_val.csv 

    # foscdata/train.csv 
    # foscdata/test.csv
    # reading either train/test pair of the above datasets;

    train_df = pd.read_csv(train_)
    train_df['labels'] = train_df.apply(onehot, axis=1)
    train_df = train_df[['payload', 'labels']]
    #train_df = train_df.sample(n=256000)

    test_df = pd.read_csv(test_)
    test_df['labels'] = test_df.apply(onehot, axis=1)
    test_df = test_df[['payload', 'labels']]
    #test_df = test_df.sample(n=32000)

    return train_df, test_df


def compute_metrics(eval_pred):

    preds, labels = eval_pred
    metrics = dict()
    preds = np.array(preds) >= 0.5
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['micro-precision'] = precision_score(labels, preds, average='micro', zero_division=0)
    metrics['macro-precision'] = precision_score(labels, preds, average='macro', zero_division=0)
    metrics['micro-recall'] = recall_score(labels, preds, average='micro', zero_division=0)
    metrics['macro-recall'] = recall_score(labels, preds, average='macro', zero_division=0)
    metrics['micro-f1'] = f1_score(labels, preds, average='micro', zero_division=0)
    metrics['macro-f1'] = f1_score(labels, preds, average='macro', zero_division=0)

    print('\n' + 'accuracy: {}'.format(metrics['accuracy']) + '\n')
    print('micro-precision: {}'.format(metrics['micro-precision']) + '.\n')
    print('macro-precision: {}'.format(metrics['macro-precision']) + '.\n')
    print('micro-recall: {}'.format(metrics['micro-recall']) + '.\n')
    print('macro-recall: {}'.format(metrics['macro-recall']) + '.\n')
    print('micro-f1: {}'.format(metrics['micro-f1']) + '.\n')
    print('macro-f1: {}'.format(metrics['macro-f1']) + '.\n')

    return metrics