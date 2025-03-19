import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import re
from pyriksdagen.utils import remove_whitespace_from_sequence
import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def precision(labels, preds, target_label):
    res = [1 if label == pred else 0 for label, pred in zip(labels, preds) if pred == target_label] 
    try:
        p = sum(res) / len(res)
    except:
        p = 0
    return p

def recall(labels, preds, target_label):
    res = [1 if label == pred else 0 for label, pred in zip(labels, preds) if label == target_label] 
    try:
        r = sum(res) / len(res)
    except:
        r = 0
    return r

def F1(pre, rec):
    try:
        f1 = 2/((1/pre)+(1/rec))
    except:
        f1 = 0
    return f1

def accuracy(labels, preds):
    res = [1 if label == pred else 0 for label, pred in zip(labels, preds)] 
    return sum(res) / len(res)

def get_metrics(labels, preds, label):
  acc = accuracy(labels, preds)
  pre = precision(labels, preds, label)
  rec = recall(labels, preds, label)
  f_1 = F1(pre, rec)
  return acc, pre, rec, f_1 

def get_split_characters(sequence):
    split_chars = []
    split_sequence_list = sequence.split('[SPLIT]')
    if len(split_sequence_list) != 1:
        n_chars = 0
        for i, sub_sequence in enumerate(split_sequence_list):
            if n_chars != 0:
                split_chars.append(n_chars)
            n_chars += len(sub_sequence)
    return split_chars

def split_tokenizer(df, tokenizer, max_length):
    input_ids, attention_masks, labels, offset_mappings = [], [], [], []
    
    for idx, row in df.iterrows():
        input_sequence = re.sub("\[SPLIT\]", "", row.text)
        encoded_dict = tokenizer.encode_plus(input_sequence,
                                             max_length = max_length,
                                             padding = 'max_length',
                                             add_special_tokens = False,
                                             return_offsets_mapping = True,
                                             truncation = True)
        split_chars = get_split_characters(row.text)
        label = []
        for offset in encoded_dict['offset_mapping']:
            if offset[0] in split_chars:
                label.append(1)
            else:
                label.append(0) 
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(label)
        offset_mappings.append(encoded_dict['offset_mapping'])
        
    output_dict = {'input_ids' : torch.tensor(input_ids),
                   'attention_masks' : torch.tensor(attention_masks),
                   'labels' : torch.tensor(labels),
                   'offset_mapping': offset_mappings}
    return output_dict

def flatten_list(list_of_lists):
    return [entry for nested_list in list_of_lists for entry in nested_list]

def evaluate(model, loader, device):
    loss = 0.0
    tokens = []
    labels, predictions = [], []
    correct_sequence_predictions = []
    model.eval()
    for batch in tqdm(loader, total = len(loader)):
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        output = model(input_ids = batch_input_ids, attention_mask = batch_attention_masks, labels = batch_labels)
        batch_predictions = torch.argmax(output.logits, -1)
        for row_idx, row in enumerate(batch_labels):
            row = row.tolist()
            row_input_ids = batch_input_ids[row_idx].tolist()
            row_attention_mask = batch_attention_masks[row_idx].tolist()
            row_predictions = batch_predictions[row_idx].tolist()
            row_labels = [entry for i, entry in enumerate(row) if row_attention_mask[i] == 1]
            row_predictions = [entry for i, entry in enumerate(row_predictions) if row_attention_mask[i] == 1]
            row_tokens = [entry for i, entry in enumerate(row_input_ids) if row_attention_mask[i] == 1]
            
            tokens.append(row_tokens)
            labels.append(row_labels)
            predictions.append(row_predictions)
            
            if row_labels == row_predictions:
                correct_sequence_prediction = 1
            else:
                correct_sequence_prediction = 0
            correct_sequence_predictions.append(correct_sequence_prediction)
        
        loss += output.loss.item()
            
    result = get_metrics(flatten_list(labels), flatten_list(predictions), 1)
    sequence_accuracy = sum(correct_sequence_predictions)/len(correct_sequence_predictions)
    
    output = {'tokens': tokens,
              'predictions': predictions,
              'result': result,
              'sequence_accuracy': sequence_accuracy,
              'loss': loss}
    return output

def split_sequences(eval_dict, df, offset_mappings):
    sequences = []
    
    for predictions, offset_mapping, sequence in zip(eval_dict['predictions'], offset_mappings, df['text']):
        split_sequence_list = []
        sequence = re.sub("\[SPLIT\]", "", sequence)
        seq_start = 0
        for prediction, offset in zip(predictions, offset_mapping):
            if prediction == 1:
                seq_end = offset[0]
                sub_sequence = sequence[seq_start:seq_end]
                split_sequence_list.append(sub_sequence)
                seq_start = seq_end
        if seq_start != len(sequence):
            sub_sequence = sequence[seq_start:len(sequence)]
            split_sequence_list.append(sub_sequence)
        sequences.append(split_sequence_list)
    return sequences

def predict_sequences(eval_dict, df, dict):
    new_sequences = split_sequences(eval_dict, df, dict['offset_mapping'])
    new_predicted_sequences = []
    for sequence in new_sequences:
        rejoined_sequence = ('[SPLIT]').join(sequence)
        new_predicted_sequences.append(rejoined_sequence)
    out_df = df
    out_df['predicted_sequence'] = new_predicted_sequences
    return out_df


batch_size = 16
n_epochs = 10
max_length = 512
learning_rate = 0.00003

def main(args):
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    f'{args.data_folder}/train_set.csv'
    df_train = pd.read_csv(f'{args.data_folder}/train_set.csv')
    df_val = pd.read_csv(f'{args.data_folder}/val_set.csv')
    df_test = pd.read_csv(f'{args.data_folder}/test_set.csv')

    df_train['text'] = [remove_whitespace_from_sequence(s) for s in df_train['text']]
    df_val['text'] = [remove_whitespace_from_sequence(s) for s in df_val['text']]
    df_test['text'] = [remove_whitespace_from_sequence(s) for s in df_test['text']]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    train_dict = split_tokenizer(df_train, tokenizer, max_length)
    val_dict = split_tokenizer(df_val, tokenizer, max_length)
    test_dict = split_tokenizer(df_test, tokenizer, max_length)

    train_dataset = TensorDataset(train_dict['input_ids'], train_dict['attention_masks'], train_dict['labels'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_dict['input_ids'], val_dict['attention_masks'], val_dict['labels'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(test_dict['input_ids'], test_dict['attention_masks'], test_dict['labels'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    id2label = {0: "other",
                1: "split_token"}
    label2id = {"other": 0,
                "split_token": 1}
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir, 
                                                            id2label = id2label, 
                                                            label2id = label2id)
    model.to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr = learning_rate)

    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        print(f'begin epoch {epoch}')
        train_loss = 0
        model.train()
        for batch in tqdm(train_loader, total = len(train_loader)):
            model.zero_grad()
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)
            output = model(input_ids = input_ids, 
                        attention_mask = attention_masks, 
                        labels = labels)
            loss = output.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print(f'epoch {epoch} done')
        val_eval_out = evaluate(model, val_loader, device)
        print(val_eval_out['result'])
        val_loss = val_eval_out['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()

    model.load_state_dict(best_model_state_dict)    

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_eval_dict = evaluate(model, train_eval_loader, device)
    print(f"training results tokens: {train_eval_dict['result']}")
    print(f"training results sequences: {train_eval_dict['sequence_accuracy']}")
    val_eval_dict = evaluate(model, val_loader, device)
    print(f"validation results: {val_eval_dict['result']}")
    print(f"val results sequences: {val_eval_dict['sequence_accuracy']}")
    test_eval_dict = evaluate(model, test_loader, device)
    print(f"test results: {test_eval_dict['result']}")
    print(f"test results sequences: {test_eval_dict['sequence_accuracy']}")

    if args.save_predictions:
        train_predictions = predict_sequences(train_eval_dict, df_train, train_dict)
        val_predictions = predict_sequences(val_eval_dict, df_val, val_dict)
        test_predictions = predict_sequences(test_eval_dict, df_test, test_dict)
        
        train_predictions.to_csv(f'{args.save_folder}/train_predictions.csv', index = False)
        val_predictions.to_csv(f'{args.save_folder}/val_predictions.csv', index = False)
        test_predictions.to_csv(f'{args.save_folder}/test_predictions.csv', index = False)
    
    model.save_pretrained(f'{args.save_folder}/split_model')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type = str, default = 'datasets/split_segments')
    parser.add_argument("--save_folder", type = str)
    parser.add_argument("--model_dir", type = str, default = 'exbert1.7')
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")
    parser.add_argument("--save_predictions", action="store_true", help="Set this flag to save predictions to csv.")
    args = parser.parse_args()
    main(args)
