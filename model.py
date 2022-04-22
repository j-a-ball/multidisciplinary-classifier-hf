import torch
import numpy as np
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)

from utils import compute_metrics


class FOSCModel:
    def __init__(self, model_ckpt, num_labels):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ckpt, 
            problem_type='multi_label_classification')

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt, 
            num_labels=num_labels, 
            problem_type='multi_label_classification')

        self.args = TrainingArguments(
            output_dir='fosc_ckpts',
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            num_train_epochs=1,
            evaluation_strategy = 'epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            load_best_model_at_end=True,
            seed=42
            )
