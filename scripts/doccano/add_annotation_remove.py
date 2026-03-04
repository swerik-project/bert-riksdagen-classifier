
""" Script for adding sentences the existing training data and removing them from the unlabeled pool 
    It takes output from Doccano, just JSONL format, and make into the JSON format of training data
    Then merges is to existing training data
    Then removes the annotated sentences from the unlabeled pool
"""

import pandas as pd
import json
import icu
import glob 
import os
import argparse
import re


### New functionality needed for the new annotation system:

# - after the last annotated sentence, drop all other remaining sentences
# - make sure add_to_train_data() only use the annotated files, but remove_sentences() use all the cases

def inception_word_tokenizer(text, locale='en_GB'):
    #Tokenize a text into words using ICU BreakIterator. Returns a List of word strings very similar to the Inception word-level tokenizer.
    bi = icu.BreakIterator.createWordInstance(icu.Locale(locale))
    bi.setText(text)
    words = []
    start = bi.first()
    for end in bi:
        word = text[start:end].strip()
        # ICU marks punctuation and spaces as separate words; filter out pure spaces
        if word:
            words.append(word)
        start = end
    return words

def update_history(jsonl_path):
    """
    Update pipeline history with the iteration number extracted from the annotated JSONL file.
    Assumes the file name for the newly annotated files contains the iteration number after an underscore, e.g., doccano_result_3.jsonl
    """

    # Extract filename
    filename = os.path.basename(jsonl_path)

    # Extract iteration number after the last underscore
    match = re.search(r'_(\d+)\.jsonl$', filename)
    if not match:
        raise ValueError(f"Could not find iteration number in filename: {filename}")

    iter_number = int(match.group(1))
    print(f"Current iteration number: {iter_number}")

    # Prepare folder and path for saving history
    history_folder = os.path.join("pipeline_history", "iteration_tracker")
    os.makedirs(history_folder, exist_ok=True)
    history_file = os.path.join(history_folder, "iteration.json")

    # Save iteration number
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({"current_iteration": iter_number}, f, indent=2)

    print(f"Iteration tracker file updated. Iteration number is now: {iter_number}")



####### Filter out the non annotated sentences from the newly annotated data 
def sentence_ids_to_remove(jsonl_path):
    
    sentences_to_drop = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            labels = row.get("label", [])
            sentence_id = row.get("meta", {}).get("sentence_id")

            # Check if any label is NOT_FULL_SENTENCE or JUNK
            if any(label_name in {"NOT_FULL_SENTENCE", "JUNK"}
                   for _, _, label_name in labels):
                sentences_to_drop.append(sentence_id)

    print(f"Number of ids that are not full sentences: {len(sentences_to_drop)}")
    return sentences_to_drop


def jsonl_to_train_format(jsonl_path, outer_dict_key, label_to_id: dict, label_to_id_final: dict):
    
    print("Making annotated sentences into json format ready to merge to existing train data")
    """
    Convert a Doccano JSONL file into tokenized sentences with token-level labels
    and group them by case_id (or other meta info) in a dictionary.
    The outer dictionary keys are numbers starting from outer_dict_key + 1 taken from the existing training file.

    Args:
        jsonl_path (str): Path to the Doccano JSONL file.
        label_to_id (dict): Mapping from label string to integer ID.
    Returns:
        Dict[int, dict]: Each key is a number, each value contains metadata and sentence info.
    """

    grouped_data = {}
    case_key_map = {}  # maps case_id to outer dict number
    next_key = outer_dict_key + 1

    with open(jsonl_path, "r", encoding="utf-8") as f:
        doccano_data = [json.loads(line) for line in f]


    #######################################################
    # HERE: remove the lines that are junk, meaning they contain annotations as "NOT_FULL_SENTENCE" or "JUNK"
    sentence_ids_drop = set(sentence_ids_to_remove(jsonl_path))
    original_len = len(doccano_data)
    doccano_data = [
        item for item in doccano_data
        if item.get("meta", {}).get("sentence_id") not in sentence_ids_drop
    ]
    print(f"Filtered out {original_len - len(doccano_data)} junk sentences")

    #######################################################

    for sid, item in enumerate(doccano_data, start=1):
        text = item["text"]
        meta = item.get("meta", {})

        # Split tokens using regex
        #tokens = re.split(r'\s+', text)

        # split tokens with Inception tokenizer emulator
        tokens = inception_word_tokenizer(text)


        # Track character offsets for token-label mapping
        offsets = []
        pos = 0
        for token in tokens:
            start = text.find(token, pos)
            end = start + len(token)
            offsets.append((start, end))
            pos = end

        # Initialize token labels to 0 (O)
        labels = [0] * len(tokens)

        # Map character-level spans to token-level labels
        for start_char, end_char, label_str in item.get("label", []):
            for i, (token_start, token_end) in enumerate(offsets):
                if token_end > start_char and token_start < end_char:
                    labels[i] = label_to_id.get(label_str, 0)

        # Determine outer dict key for this case
        case_id = meta.get("case_id", f"unknown_case_{sid}")
        if case_id not in case_key_map:
            case_key_map[case_id] = next_key
            grouped_data[next_key] = {
                "casename": case_id,  # or adjust if you want a separate casename
                "annotator": meta.get("annotator"),
                "year": meta.get("year"),
                "sentences": []
            }
            next_key += 1

        #################################################################
        # adjust the labels so that the extra NO_REFERENCE is removed.

        # Append sentence info
        outer_key = case_key_map[case_id]
        grouped_data[outer_key]["sentences"].append({
            "sid": str(sid),
            "tokens": tokens,
            "cog_tok_labels": labels,
            "cog_label": "REFERENCE" if any(l > 0 for l in labels) else "NONE",
            "cog_label_detailed": {str(v): k for k, v in label_to_id_final.items()} # using the "final" here to match existing train data which doesnt have "NO_REFERENCE"
        })

    return grouped_data



###### get largest key in the existing training file #####
def get_last_outer_key(train_file_path):

    with open(train_file_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Get the last outer dict key as an integer
    outer_dict_key = max(int(k) for k in train_data.keys())
    print("Last outer dict key in training file:", outer_dict_key)

    return outer_dict_key



"""" MERGE THE EXISTING TRAINING DATA WITH THE NEWLY ANNOTATED SENTENCES """
def merge_to_existing_train(train_file_path, new_data):

    # read in the training data
    with open(train_file_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)


    # check to ensure no key collisions
    overlapping = set(train_data.keys()) & set(new_data.keys())
    if overlapping:
        raise ValueError(f"Error: keys already exist in training data: {overlapping}")

    # Merge
    merged = {**train_data, **new_data}

    # save
    output_path  = train_file_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Successfully appended annotated data to existing training data: {output_path}")
    
    # calculating the number of newly added sentences
    num_sentences = sum(
    len(case_data["sentences"])
    for case_data in new_data.values()
    )

    print(f"Added {num_sentences} new training sentences")



""" Remove these senteces from the unlabeled pool so that they can not be selected again """
def remove_sentences(input_folder,jsonl_path):

    # get sentence IDs of the newly annotated examples
    sentence_ids = []
    with open(jsonl_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            # Adjust the key depending on your file structure:
            sentence_ids.append(obj["meta"]["sentence_id"])      # or obj["sentence_id"], etc.

    print(f'Removing {len(sentence_ids)} sentences from the unlabeled pool')


    # Find parquet files
    sentence_files = glob.glob(os.path.join(input_folder, "sentences*.parquet"))
    all_files = sentence_files

    print("Found files:", len(all_files))
    print("Starting processing parquet files. May take 1-2 min to read an write all of them")

    rows_removed_count = 0
    for file in all_files:
        
        df = pd.read_parquet(file)

        # Make sure the column exists
        if "sentence_id" not in df.columns:
            print(f"  Skipped (no 'sentence_id' column): {file}")
            continue

        before = len(df)
        df = df[~df["sentence_id"].isin(sentence_ids)]
        after = len(df)

        rows_removed_count += (before-after)

        # Overwrite file with cleaned version
        df.to_parquet(file, index=False)    

    print(f'Successfully removed {rows_removed_count} rows from the unlabeled pool')


def main(args):
    
    print("Training data:", args.train_file)
    print("Latest annotations:", args.annotations_file)
    print("Unlabeled pool is in:", args.unlabeled_pool)
    
    label_to_id = {
        "O": 0,
        "NO_REFERENCE": 0,
        "STATUTE_REFERENCE": 1,
        "UK_STATUTE_REFERENCE": 2,
        "CASE_REFERENCE": 3,
        "UK_CASE_REFERENCE": 4,
        "CASE_NEUTRAL_CITATION": 5,
        "CASE_NAME_PARTIES": 6,
        "CASE_LAW_REPORT_REFERENCE": 7
    }

    # this the labels without the NO_REFERENCE which is just for annotation purpuses but isnt used in the resulting training data.
    label_to_id_final = {
        "O": 0,
        "STATUTE_REFERENCE": 1,
        "UK_STATUTE_REFERENCE": 2,
        "CASE_REFERENCE": 3,
        "UK_CASE_REFERENCE": 4,
        "CASE_NEUTRAL_CITATION": 5,
        "CASE_NAME_PARTIES": 6,
        "CASE_LAW_REPORT_REFERENCE": 7
    }

    outer_dict_key = get_last_outer_key(args.train_file)
    
    new_data = jsonl_to_train_format(args.annotations_file, outer_dict_key, label_to_id, label_to_id_final) # new_data is annotated data ready to merge onto existing training data.

    # Just save for manual inspection
    with open("new_sentences_debug.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print("Saved new_sentences_debug.json for manual format inspection")

    # update iteration tracker in json file "tracker.json"
    update_history(args.annotations_file)

    # calling the merge function. No output from it is needed
    merge_to_existing_train(args.train_file, new_data)

    # calling the remove function to remove the annotated examples from the unlabeled pool.
    remove_sentences(args.unlabeled_pool, args.annotations_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add annotated sentences to training data and remove from unlabeled pool.")
    parser.add_argument("--train_file", default="./data/splits/train_110_AL.json", type=str, help="path to training json format training data up to last iteration of AL")
    parser.add_argument("--annotations_file", default="./doccano_files/doccano_results_iter_XXX.jsonl", type=str,  help="Path to jsonl file with the latest annotated cases")
    parser.add_argument("--unlabeled_pool", default="unlabeled_pool", type=str, help="folder were the unlabeled pool (parquet files with sentences) is")

    args = parser.parse_args()
    main(args)