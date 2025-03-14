from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from pyriksdagen.utils import protocol_iterators
from pyparlaclarin.refine import reclassify, format_texts

# others
import argparse
from lxml import etree
import progressbar

def main(args):
    
    # redefining labels in pretrained model to work with reclassify()
    id2label = {0 : 'note',
                1 : 'u'}
    label2id = {'note' : 0,
                'u' : 1}

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu' 

    model = AutoModelForSequenceClassification.from_pretrained(args.model_folder, id2label = id2label, label2id = label2id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_folder)

    pipe = pipeline(task = 'text-classification', model = model, tokenizer = tokenizer, device = device,
                    max_length = 512, truncation = True, padding = 'max_length')
    classifier = lambda elem: pipe(elem.text)[0]['label']

    protocol_list = list(protocol_iterators(args.records_folder, start=args.start, end=args.end))
    protocols = sorted(protocol_list)

    for protocol_path in progressbar.progressbar(protocols):
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.parse(protocol_path, parser).getroot()

        root = reclassify(root, classifier, exclude=["date", "speaker"])
        root = format_texts(root, padding = 10)
        b = etree.tostring(root, pretty_print=True, encoding="utf-8", xml_declaration=True)
        
        with open(protocol_path, "wb") as f:
            f.write(b)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start", type=int, default=1867, help="Start year")
    parser.add_argument("-e", "--end", type=int, default=2023, help="End year")
    parser.add_argument("--model_folder", type = str)
    parser.add_argument("--records_folder", type = str)
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")
    args = parser.parse_args()
    main(args)