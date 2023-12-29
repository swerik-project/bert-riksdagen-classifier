import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from transformers import get_linear_schedule_with_warmup
import argparse
from tqdm import tqdm
import os
from bidict import bidict

# Logger go brrr pretty colors !
# Don't mind me
import sys, logging, colorlog
TRAIN = 25
LOG_COLORS = {'DEBUG':'cyan', 'INFO':'green', 'TRAIN':'blue', 'WARNING':'yellow', 'ERROR': 'red', 'CRITICAL':'red,bg_white'}
logging.addLevelName(TRAIN, 'TRAIN')
HANDLER = colorlog.StreamHandler(stream=sys.stdout)
HANDLER.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(asctime)s [%(levelname)s] %(white)s(%(name)s)%(reset)s: %(message)s',
                                               log_colors=LOG_COLORS,
                                               datefmt="%H:%M:%S",
                                               stream=sys.stdout))
LOGGER = colorlog.getLogger('finetuner')
LOGGER.addHandler(HANDLER)
LOGGER.setLevel(10)
# Done with colors going brr


def encode(df, tokenizer):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for ix, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
                            row['content'],                      
                            add_special_tokens = True,
                            max_length = 512,
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['tag'].tolist())

    return input_ids, attention_masks, labels


def evaluate(model, loader, bayesian_bootstrap=False):
    loss, accuracy = None, None
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(args.device)
        input_mask = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        output = model(input_ids,
            token_type_ids=None, 
            attention_mask=input_mask, 
            labels=labels)
        loss_batch = torch.nn.functional.cross_entropy(output.logits, labels, reduction="none").detach().cpu()
        preds_batch = torch.argmax(output.logits, axis=1)
        batch_acc = (preds_batch == labels).float().cpu()
        if loss is None:
            loss = loss_batch
            accuracy = batch_acc
        else:
            loss = torch.cat([loss, loss_batch])
            accuracy = torch.cat([accuracy, batch_acc])
    
    if not bayesian_bootstrap:
        accuracy = accuracy.numpy()
        loss = loss.numpy()
    else:
        np.random.seed(429)
        N = len(accuracy)
        theta = np.random.dirichlet(np.ones(N), 1000)
        accuracy = theta @ accuracy.numpy()
        loss = theta @ loss.numpy()
    return loss, accuracy


def main(args): 
    df = pd.read_csv(f'{args.data_path}')
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Create binary label where seg = 1
    df = df[df["content"].notnull()]
    label_names = args.label_names
    if label_names is None:
        label_names = sorted(list(set(df["tag"])))
    label_dict = {ix: name for ix, name in enumerate(label_names)}
    df["tag"] = [bidict(label_dict).inv[tag] for tag in df["tag"]]

    LOGGER.info("Load and save tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.savepath is not None:
        tokenizer.save_pretrained(args.savepath)

    
    LOGGER.info("Preprocess datasets...")
    input_ids, attention_masks, labels = encode(df, tokenizer)

    LOGGER.info(f"Labels: {labels}")

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size  = int(args.train_ratio * len(dataset))
    val_size    = int(args.valid_ratio * len(dataset))
    test_size   = len(dataset) - train_size - val_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )

    valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )

    # Not used atm
    test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )

    LOGGER.info("Define model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(label_dict),
        id2label=label_dict).to(args.device)

    # Initialize optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.n_epochs
    num_warmup_steps = num_training_steps // 10

    # Linear warmup and step decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
        )


    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for epoch in range(args.n_epochs):
        LOGGER.log(TRAIN, f"Epoch {epoch} starts!")
        train_loss = 0
        model.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            model.zero_grad()   

            input_ids = batch[0].to(args.device)
            input_mask = batch[1].to(args.device)
            labels = batch[2].to(args.device)
            output = model(input_ids,
                token_type_ids=None, 
                attention_mask=input_mask, 
                labels=labels)
            loss = output.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
                    
        # Evaluation
        valid_loss, valid_accuracy = evaluate(model, valid_loader, bayesian_bootstrap=args.bayesian_bootstrap)

        train_losses.append(train_loss)
        valid_losses.append(torch.tensor(valid_loss))

        train_loss_avg = train_loss * args.batch_size / len(train_loader)
        valid_loss_avg = np.mean(valid_loss)
        valid_accuracy_avg = np.mean(valid_accuracy)

        LOGGER.log(TRAIN, f'Training Loss: {train_loss_avg:.3f}')
        LOGGER.log(TRAIN, f'Validation Loss: {valid_loss_avg:.3f}')
        LOGGER.log(TRAIN, f'Validation accuracy: {valid_accuracy_avg}')

        if args.bayesian_bootstrap:
            valid_losses_bbs = torch.stack(valid_losses, dim=0)
            valid_losses_bbs = torch.argmax(-valid_losses_bbs, axis=0)
            
            bincounts = torch.bincount(valid_losses_bbs)
            posterior_probs = bincounts / bincounts.sum()
            LOGGER.log(TRAIN, f"Bayesian bootstrap samples: {bincounts}")
            LOGGER.log(TRAIN, f"Posterior probabilities for best model: {posterior_probs}")

        if valid_loss_avg < best_valid_loss:
            LOGGER.info("Best validation loss so far")
            best_valid_loss = valid_loss_avg
            if args.savepath is not None:
                LOGGER.debug(f"Save model to {args.savepath}")
                model.save_pretrained(args.savepath)
            else:
                LOGGER.debug("No save path provided, skipping saving...")
        else:
            LOGGER.info("Not the best validation loss so far")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="KBLab/bert-base-swedish-cased")
    parser.add_argument("--tokenizer", type=str, default="KBLab/bert-base-swedish-cased")
    parser.add_argument("--label_names", type=str, nargs="+", default=None)
    parser.add_argument("--data_path", type=str, default="data/training_data.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.00002)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.2) # test set is what remains after train and valid splits
    parser.add_argument("--bayesian_bootstrap", type=bool, default=False)
    args = parser.parse_args()
    main(args)
