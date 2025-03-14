from pyriksdagen.utils import infer_metadata, XML_NS, remove_whitespace_from_sequence
from alto import parse_file
from alto import String as altoString
from thefuzz import fuzz
import re

#### functions to get position features from xml files ####

def relative_page_number(x, max_x):
    try:
        output = (x/max_x)*1000
    except:
        output = 0.0
    return output

def get_max_index(l, elem):
    """
    If there are identical text sequences on a page, returns index of sequence which is closest in sequential order
    """
    maxval = None
    for i, val in enumerate(l):
        if maxval is None or val > maxval:
            indices = [i]
            maxval = val
        elif val == maxval:
            indices.append(i)
    if len(indices) == 1:
        return indices[0]
    else:
        previous_elements = elem.xpath("preceding::*[local-name() = 'note' or local-name() = 'seg' or local-name() = 'pb']")
        for i, e in enumerate(reversed(previous_elements)):
            if e.tag[-2:] == 'pb':
                page_position = i
                break
        return min(indices, key = lambda x: abs(x - page_position))

def merge_dictionaries(dict1, dict2):
    output_dict = {**dict1, **dict2}
    return output_dict

def get_img_box(input_box):
    # returns corner coordinates of img as tuple
    input_box = [float(coord) for coord in input_box]
    output = (input_box[0], input_box[1], input_box[0] + input_box[2], input_box[1] + input_box[3])
    return output

def update_block_position(prev_pos, curr_pos):
    output_0 = min(prev_pos[0], curr_pos[0])
    output_1 = min(prev_pos[1], curr_pos[1])
    output_2 = max(prev_pos[2], curr_pos[2])
    output_3 = max(prev_pos[3], curr_pos[3])
    
    return (output_0, output_1, output_2, output_3)

def get_protocol_information(elem):
    sequence_id = elem.get(f'{XML_NS}id')
    head_elem = elem.xpath("preceding::*[local-name() = 'head'][1]")[0]
    pkg_name = head_elem.text
    metadata = infer_metadata(pkg_name)
    record_year = metadata['year']
    record_chamber = metadata['chamber']
    page_elem = elem.xpath("preceding::*[local-name() = 'pb'][1]")[0]
    last_page_elem_list = elem.xpath("following::*[local-name() = 'pb']")
    last_page_elem = last_page_elem_list[-1] if len(last_page_elem_list) != 0 else None
    try:
        page_number_as_str = page_elem.get('facs').split('.')[-2][-3:]
        page_number = int(page_number_as_str)
        protocol_length_as_str = last_page_elem.get('facs').split('.')[-2][-3:] if last_page_elem is not None else page_number_as_str
    except:
        page_number_as_str = page_elem.get('facs').split('=')[-1]
        page_number = int(page_number_as_str)
        protocol_length_as_str = last_page_elem.get('facs').split('=')[-1] if last_page_elem is not None else page_number_as_str
    even_page = 1 if page_number % 2 == 0 else 0
    
    output_dict = {'id': sequence_id, 
                   'page_number' : page_number,
                   'pn_as_str' : page_number_as_str,
                   'relative_page_number' : relative_page_number(page_number, int(protocol_length_as_str)),
                   'even_page' : even_page,
                   'record_length' : int(protocol_length_as_str),
                   'year' : record_year,
                   'chamber' : record_chamber,
                   'second_chamber' : 1 if record_chamber == 'Andra kammaren' else 0,
                   'unicameral' : 1 if record_chamber == 'Enkammarriksdagen' else 0,
                   'pkg_name': pkg_name}
    return output_dict

def generate_strings_from_alto_block(alto_block):
    return (string for text_line in alto_block.text_lines 
            for string in text_line.strings 
            if isinstance(string, altoString))

def get_page_position_information(elem, alto, elem_type = None):
    text_sequence = remove_whitespace_from_sequence(elem.text)
    page_info = alto.layout.pages[0]
    page_width, page_height = page_info.width, page_info.height
    text_blocks = []
    alto_blocks = alto.extract_text_blocks()
    text_blocks = [' '.join(text_block) for text_block in alto.extract_grouped_words(group_by = 'TextBlock')]
    
    if elem_type == 'speaker':
        n_colons = text_sequence.count(':')
        text_blocks = [''.join(re.split('(:)', x)[:(n_colons+1)]) for x in text_blocks]
        dist_ratio_list = [fuzz.ratio(text_sequence, x) for x in text_blocks]
        max_index = get_max_index(dist_ratio_list, elem)
        matched_block = alto_blocks[max_index]
        curr_pos = [9999.0, 9999.0, 0.0, 0.0]
        n_passed_colons = 0
        for string in generate_strings_from_alto_block(matched_block):
            curr_word = string.content
            word_pos = get_img_box([string.hpos, string.vpos, string.width, string.height])
            curr_pos = update_block_position(curr_pos, word_pos)
            if ':' in curr_word:
                n_passed_colons += 1
                if n_passed_colons == n_colons:
                    break
        block_pos = curr_pos
    else:
        dist_ratio_list = [fuzz.ratio(text_sequence, x) for x in text_blocks]
        max_index = get_max_index(dist_ratio_list, elem)
        matched_block = alto_blocks[max_index]
        block_pos = get_img_box([matched_block.hpos, matched_block.vpos, matched_block.width, matched_block.height])

    output_dict = {'page_width' : page_width,
                   'page_height' : page_height,
                   'actualposLeft' : block_pos[0],
                   'actualposUpper' : block_pos[1],
                   'actualposRight' : block_pos[2],
                   'actualposLower' : block_pos[3],
                   'posLeft' : (block_pos[0]/page_width)*1000,
                   'posUpper' : (block_pos[1]/page_height)*1000,
                   'posRight' : (block_pos[2]/page_width)*1000,
                   'posLower' : (block_pos[3]/page_height)*1000}
    return output_dict

def page_as_string(page_number):
    # changes page number to string format used in kblab database
    return f"{page_number:0>3}"

def get_elem_data(elem, archive, elem_type = None, page_offset = 0):
    out = get_protocol_information(elem)
    pkg_name = out['pkg_name']
    page_not_found = True
    while page_not_found:
        pn_as_str = page_as_string(out['page_number'] + page_offset)
        xml_id = pkg_name.replace('-', '_') + '-' + pn_as_str + '.xml'
        try:
            pkg = archive.get(pkg_name)
            alto = parse_file(pkg.get_raw(xml_id))
            out_pos = get_page_position_information(elem, alto, elem_type = elem_type)
            page_not_found = False
        except:
            page_offset += 1
            if page_offset >= 11:
                out_pos = {'page_width' : None,
                            'page_height' : None,
                            'posLeft' : None,
                            'posUpper' : None,
                            'posRight' : None,
                            'posLower' : None}
                page_not_found = False
            
    merged_output = merge_dictionaries(out, out_pos)
    return merged_output, page_offset
