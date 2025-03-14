"""
Find sequences to join in protocols using BERT models
"""
# transformers
from transformers import pipeline

from pyriksdagen.utils import protocol_iterators, elem_iter, get_sequence_from_elem_list, remove_whitespace_from_sequence
from pyparlaclarin.refine import format_texts

# others
import argparse
import os
from lxml import etree
import progressbar

def join_elems(root, classifier):
    for tag, elem in elem_iter(root):
        if tag == 'note':
            prev_elem_list = elem.xpath("preceding::*[local-name() = 'note' or local-name() = 'seg' or local-name() = 'pb'][1]")
            if len(prev_elem_list) > 0:
                if prev_elem_list[0].tag[-2:] == 'pb':
                    pass
                else:
                    prev_sequence = get_sequence_from_elem_list(prev_elem_list)
                    new_sequence = remove_whitespace_from_sequence(prev_sequence + elem.text)
                    c = classifier(new_sequence)
                    if c == 'join':
                        prev_elem_list[0].text = new_sequence
                        elem.getparent().remove(elem)
                
        elif tag == 'u':
            for child in elem:
                prev_elem_list = elem.xpath("preceding::*[local-name() = 'note' or local-name() = 'seg' or local-name() = 'pb'][1]")
                if len(prev_elem_list) > 0:
                    if prev_elem_list[0].tag[-2:] == 'pb':
                        pass
                    else:
                        prev_sequence = get_sequence_from_elem_list(prev_elem_list)
                        new_sequence = remove_whitespace_from_sequence(prev_sequence + child.text)
                        c = classifier(new_sequence)
                        if c == 'join':
                            prev_elem_list[0].text = new_sequence
                            child.getparent().remove(child)
        
    return root

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    # Initialize classifier
    pipe = pipeline(task = 'text-classification', model = 'fberi/BertModel-join', max_length = 512, truncation = True, device = device)
    classifier = lambda text: pipe(text)[0]['label']

    
    protocol_list = list(protocol_iterators(args.records_folder, start=args.start, end=args.end))
    protocols = sorted(protocol_list)

    for protocol_path in progressbar.progressbar(protocols):

        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(protocol_path, parser)
        root = tree.getroot()
        
        root = join_elems(root, classifier)
        root = format_texts(root, padding = 10)
        b = etree.tostring(root, pretty_print=True, encoding="utf-8", xml_declaration=True)
    
        with open(protocol_path, "wb") as f:
            f.write(b)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2023, help="End year")
    parser.add_argument("--records_folder", type = str)
    parser.add_argument("--save_folder", type = str, default = None)
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")
    args = parser.parse_args()
    main(args)