"""
Find title sequences in protocols using BERT models
"""
# transformers
from transformers import AutoTokenizer, pipeline

from pyriksdagen.utils import protocol_iterators, get_context_sequences_for_protocol, elem_iter, XML_NS

# others
import argparse
import os
from lxml import etree
import progressbar

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

    
    protocol_list = list(protocol_iterators(args.records_folder, start=args.start, end=args.end))
    protocols = sorted(protocol_list)

    for protocol in progressbar.progressbar(protocols):
        context_dict = get_context_sequences_for_protocol(protocol, args.context_type)
        
        if len(context_dict['id']) != 0:
            
            prediction_list = context_pipe(context_dict['text'])
            prediction_dict = {}
            for i, seq_id in enumerate(context_dict['id']):
                prediction_dict[seq_id] = (prediction_list[i]['label'], prediction_list[i]['score'])
            
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(protocol, parser)
            root = tree.getroot()
            for tag, elem in elem_iter(root):
                if tag == 'note':
                    curr_idx = elem.attrib[f"{XML_NS}id"]
                    curr_pred = prediction_dict[curr_idx][0]
                        
                    if curr_pred == 'title':
                        elem.attrib['type'] = 'title'
                    
                elif tag == 'u':
                    for child in elem.getchildren():
                        curr_idx = child.attrib[f"{XML_NS}id"]
                        curr_pred = prediction_dict[curr_idx][0]
                        
                        if curr_pred == 'title':
                            child.attrib['type'] = 'title'
                        
            f = open(protocol, 'wb')
            f.write(etree.tostring(tree, pretty_print = True, encoding = 'utf-8', xml_declaration = True))
            f.close()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2023, help="End year")
    parser.add_argument("--context_type", default = None, type=str)
    parser.add_argument("--records_folder", type = str)
    parser.add_argument("--save_folder", type = str, default = None)
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")
    args = parser.parse_args()
    main(args)