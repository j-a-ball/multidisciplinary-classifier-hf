import numpy as np
import pandas as pd
import torch
import transformers
print(f"Running on transformers v{transformers.__version__}")

from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import FOSCDataset
from model import FOSCModel
from utils import onehot, fetch_data, compute_metrics


#trainer
class FOSCTrainer:
    def __init__(self, model_, args_, tokenizer_, train_, test_):
        self.trainer = Trainer(
            model_,
            args_,
            tokenizer=tokenizer_,
            train_dataset=train_,
            eval_dataset=test_,
            compute_metrics=compute_metrics
            )


#run
if __name__ == "__main__":

    #gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #load data
    train, val = fetch_data('fosc_data/train_train.csv', 'fosc_data/train_val.csv')
    print('Data loaded successfully.')

    #AutoModelForSequenceClassification from model.py
    fosc_m = FOSCModel(
        model_ckpt='./fosc_ckpts/checkpoint-3858', 
        num_labels=20,
        )
    print('FOSCModel instantiated.')

    #AutoTokenizer from model.py
    tokenizer = fosc_m.tokenizer
    print('Tokenizer instantiated.')

    #train dataset
    train_encodings = tokenizer(train['payload'].values.tolist(), truncation=True, max_length=200)
    train_labels = train['labels'].values.tolist()
    #test dataset
    test_encodings = tokenizer(val['payload'].values.tolist(), truncation=True, max_length=200)
    test_labels = val['labels'].values.tolist()

    #custom dataloaders from dataset.py
    train_dataset = FOSCDataset(train_encodings, train_labels)
    test_dataset = FOSCDataset(test_encodings, test_labels)
    print('Dataloaders instantiated.')

    #model
    model = fosc_m.model.to(device)
    print('Model instantiated.')

    #args
    args = fosc_m.args

    print('Trainer arguments set.')

    #trainer
    fosc_t = FOSCTrainer(
        model_=model,
        args_=args,
        tokenizer_=tokenizer,
        train_=train_dataset,
        test_=test_dataset
        )

    trainer = fosc_t.trainer
    print('Trainer now running:')

    #train
    trainer.train()

    #val
    trainer.evaluate()
