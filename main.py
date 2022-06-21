
from argparse import ArgumentParser
import cv2
from tqdm import tqdm
import json
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import nn
from torch.optim import AdamW

from dataset import MERecognitionDataset
from model import MERecognitionModel

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("-t", "--task", choices=["train", "eval"], default="train")
    parser.add_argument("-d", "--data_dir", type=str, default="data/")
    parser.add_argument("-i", "--image_dir", type=str, default="data/data_png_Training/")
    parser.add_argument("-m", "--model_path", type=str, default="trained_model/45.pt")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-s", "--sequence_length", type=int, default=64)
    parser.add_argument("-sd", "--save_dir", type=str, default="trained_model/")
    parser.add_argument("-p", "--patience", type=int, default=5)
    
    return parser.parse_args()

def eval(args):
    
    data_dir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
    image_dir = args.image_dir if args.image_dir.endswith('/') else args.image_dir + '/'
    save_dir = args.save_dir if args.save_dir.endswith('/') else args.save_dir + '/'

    transform = A.Compose([
        A.Resize(224, 224),
        A.RandomScale(scale_limit=(-0.5, 2.0), p=.5),
        A.Normalize(),
        ToTensorV2(),
    ])

    eval_set = MERecognitionDataset(
        data_path = data_dir + "eval.json",
        image_dir = image_dir,
        transform = transform,
        sequence_length = args.sequence_length,
    )

    eval_loader = eval_set.get_dataloader(batch_size = args.batch_size, shuffle = False)

    vocab_size = max([max([ord(char) for char in line]) for line in open(data_dir + "labels.txt").readlines()]) + 1

    model = MERecognitionModel(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(args.model_path))

    model.eval()
    predictions = []
    index = 0

    exp_rate = {"exp_rate": 0, "<=1": 0, "<=2": 0, "<=3": 0}

    for batch_index, batch in enumerate(eval_loader):

        images = batch["images"].to(device)
        label_ids = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        decoder_input_ids = label_ids[:, :-1]
        label_ids = label_ids[:, 1:]
        attention_mask = attention_mask[:, :-1]

        with torch.no_grad():
            output = model.generate(images, decoder_input_ids)

        for predict, label in zip(output, label_ids):
            predict = predict.argmax(-1)

            acc = (predict == label).sum() / len(predict)

            predict = predict[:predict.tolist().index(3) if 3 in predict else len(predict)]
            label = label[:label.tolist().index(3) if 3 in label else len(label)]

            predictions.append({
                "i": index,
                "image": eval_set.data[index]["images"],
                "p": "".join([chr(p) for p in predict]),
                "l": "".join([chr(l) for l in label])
            })

            if 1 - acc < 0.03:
                exp_rate["<=3"] += 1

            if 1 - acc < 0.02:
                exp_rate["<=2"] += 1

            if 1 - acc < 0.01:
                exp_rate["<=1"] += 1

            if acc == 1:
                exp_rate["exp_rate"] += 1

            index += 1

    with open("predictions.json", 'w') as f:
        json.dump(predictions, f, ensure_ascii=False, indent='\t')

    for k in exp_rate:
        exp_rate[k] /= index - 1

    print(exp_rate)

def train(args):
    
    data_dir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
    image_dir = args.image_dir if args.image_dir.endswith('/') else args.image_dir + '/'
    save_dir = args.save_dir if args.save_dir.endswith('/') else args.save_dir + '/'

    transform = A.Compose([
        A.RandomScale(scale_limit=(-0.5, 2.0), p=1.),
        A.PadIfNeeded(min_height=256, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
        A.Resize(256, 1024),
        A.Normalize(),
        ToTensorV2(),
    ])

    train_set = MERecognitionDataset(
        data_path = data_dir + "train.json",
        image_dir = image_dir,
        transform = transform,
        sequence_length = args.sequence_length,
    )

    train_loader = train_set.get_dataloader(batch_size = args.batch_size, shuffle = True)

    eval_set = MERecognitionDataset(
        data_path = data_dir + "eval.json",
        image_dir = image_dir,
        transform = transform,
        sequence_length = args.sequence_length,
    )

    eval_loader = eval_set.get_dataloader(batch_size = args.batch_size, shuffle = False)

    vocab_size = max([max([ord(char) for char in line]) for line in open(data_dir + "labels.txt").readlines()]) + 1

    model = MERecognitionModel(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    best_eval_acc = 0.

    patience = 0

    for epoch in range(10000):

        model.train()
        train_loss = 0.

        tqdm_train_loader= tqdm(train_loader)

        for batch_index, batch in enumerate(tqdm_train_loader):

            optimizer.zero_grad()

            images = batch["images"].to(device)
            label_ids = batch["label_ids"].to(device)

            decoder_input_ids = label_ids[:, :-1]
            label_ids = label_ids[:, 1:]

            output = model(images, decoder_input_ids)

            output = output.reshape(-1, output.size(-1))
            label_ids = label_ids.reshape(-1)

            loss = criterion(output, label_ids)
            train_loss += loss.item()
            tqdm_train_loader.set_description(
                "epoch-{}_loss-{}".format(epoch + round(batch_index / len(train_loader), 2), round(loss.item(), 4))
            )

            loss.backward()
            optimizer.step()

        print("[{}] train_loss: {}".format(epoch, train_loss / len(train_loader)))

        model.eval()

        eval_acc = 0
        eval_total = 0
        index = 0
        predictions = []

        for batch_index, batch in enumerate(tqdm(eval_loader)):

            images = batch["images"].to(device)
            label_ids = batch["label_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            decoder_input_ids = label_ids[:, :-1]
            label_ids = label_ids[:, 1:]
            attention_mask = attention_mask[:, :-1]

            with torch.no_grad():
                output = model(images, decoder_input_ids, attention_mask=attention_mask)

            for predict, label in zip(output, label_ids):
                predict = predict.argmax(-1)
                pad_index = label.tolist().index(0) if 0 in label else len(label)

                predict = predict[:pad_index]
                label = label[:pad_index]

                predictions.append({
                    "i": index,
                    "image": eval_set.data[index]["images"],
                    "p": "".join([chr(p) for p in predict]),
                    "l": "".join([chr(l) for l in label])
                })
                
                eval_acc += (predict == label).sum()
                eval_total += pad_index

                index += 1
            
        eval_acc = eval_acc / eval_total

        print("[{}] eval_acc: {}".format(epoch, eval_acc))

        if eval_acc > best_eval_acc:
            
            with open("temp-predictions.json", 'w') as f:
                json.dump(predictions, f, ensure_ascii=False, indent='\t')

            if len(os.listdir(save_dir)) == 1:
                os.remove(save_dir + os.listdir(save_dir)[0])

            torch.save(model.state_dict(), save_dir + "{}.pt".format(epoch))

            best_eval_acc = eval_acc
            patience = 0

        else:
            patience += 1

            if patience == args.patience:
                break

            model.load_state_dict(torch.load(save_dir + os.listdir(save_dir)[0]))

if __name__=="__main__":

    args = parse_args()

    if args.task == "train":
        train(args)
    elif args.task == "eval":
        eval(args)