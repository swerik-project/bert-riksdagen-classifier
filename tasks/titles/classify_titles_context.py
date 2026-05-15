"""
Find title sequences in protocols using BERT models
"""
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.utils import (
    elem_iter, 
    get_context_sequences_for_protocol, 
    parse_tei,
    write_tei,
)
from transformers import (
    AutoTokenizer, 
    pipeline,
)
import argparse
import os
from lxml import etree
import tqdm




def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # loading pre-trained bert-model used in position model with tokenizer  
    tokenizer = AutoTokenizer.from_pretrained('fberi/BertModel-lc')
    # Initialize model pipeline
    context_pipe = pipeline(task = 'text-classification', 
                             model = 'fberi/BertModel-lc', 
                             tokenizer = tokenizer, 
                             device = 'cuda', 
                             trust_remote_code = True,
                             max_length = 128, truncation = True, padding = 'max_length', 
                             batch_size = 128) 
    
    for protocol in tqdm(args.records):
        context_dict = get_context_sequences_for_protocol(protocol, args.context_type)
        if len(context_dict['id']) != 0:
            prediction_list = context_pipe(context_dict['text'])
            prediction_dict = {}
            for i, seq_id in enumerate(context_dict['id']):
                prediction_dict[seq_id] = (prediction_list[i]['label'], prediction_list[i]['score'])
            
            root, ns = parse_tei(protocol)

            for tag, elem in elem_iter(root):
                if tag == 'note':
                    curr_idx = elem.attrib[f"{ns['xml_ns']}id"]
                    curr_pred = prediction_dict[curr_idx][0]
                    if curr_pred == 'title':
                        elem.attrib['type'] = 'title'
                elif tag == 'u':
                    for child in elem.getchildren():
                        curr_idx = child.attrib[f"{ns['xml_ns']}id"]
                        curr_pred = prediction_dict[curr_idx][0]
                        
                        if curr_pred == 'title':
                            child.attrib['type'] = 'title'
            write_tei(root, protocol)



        
if __name__ == "__main__":
    parser = fetch_parser("records", docstring=__doc__)
    #parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    #parser.add_argument("-e", "--end", type=int, default=2023, help="End year")
    parser.add_argument("--context_type", default = None, type=str)
    #parser.add_argument("--records_folder", type = str)
    #parser.add_argument("--save_folder", type = str, default = None)
    #parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")
    main(impute_args(parser.parse_args()))
