from sklearn.metrics import accuracy_score
import random

import torch

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

#이것들만 구현해서 모듈을 불러온 것
from bert_data_loader import TextClassificationCollator, TextClassificationDataset

import argparse

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    # Recommended model list:
    # - kykim/bert-kor-base
    # - kykim/albert-kor-base
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')
    p.add_argument('--use_albert', action='store_true')

    p.add_argument('--valid_ratio', type=float, default=.2)
    #멀티 GPU라서 이렇게 함, GPU가 1개면 그냥 돌려도 됨
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--warmup_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config

def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                label, text = line.strip().split('\t') #tsv 파일, 첫번째 column이 label, 두번째가 text
                labels += [label]
                texts += [text]

    return labels, texts

def get_datasets(
    fn, #config.train_fn 값이 들어감
    valid_ratio=.2
    ):
     # Get list of labels and list of texts.
    labels, texts = read_text(fn)

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    train_dataset = TextClassificationDataset(texts[:idx], labels[:idx])
    valid_dataset = TextClassificationDataset(texts[idx:], labels[idx:])

    return train_dataset, valid_dataset, index_to_label


def main(config):
    # Get pretrained tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name) #default='beomi/kcbert-base'
    # Get datasets and index to label map.
    train_dataset, valid_dataset, index_to_label = get_datasets(
        config.train_fn,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count() #device 갯수만큼
    n_total_iterations = int(
        #한 에폭당 train 갯수
        len(train_dataset) / total_batch_size * config.n_epochs
        )
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name, #알아서 다운받음
        num_labels=len(index_to_label)
    )

    #원래는 다 Trainer를 구현해야하지만, hugging face에서 쉽게 구현함
    training_args = TrainingArguments(
        output_dir='./.checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True, #hard precision을 쓸거냐?, 2000번때 GPU부터는 속도가 빨라짐, 하면 좋음
        evaluation_strategy='epoch',
        save_strategy='epoch',
        #logging_steps=n_total_iterations // 100,
        logging_steps=n_total_iterations,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )

    #accuracy를 계산하는 함수
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return {
            #sklearn에서 accuracy_score를 사용
            'accuracy': accuracy_score(labels, preds)
        }

    #trainer 사용
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextClassificationCollator(tokenizer,
                                       config.max_length,
                                       with_text=False),
        #dataloader가 안에 들어있어서 바로 넣으면 됨
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics, #바로 위에 있는 함수를 사용
    )

    trainer.train()

    #모델 경로 설정
    model_path = './models/' + config.model_fn

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': trainer.model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, model_path)


if __name__ == '__main__':
    config = define_argparser()
    main(config)