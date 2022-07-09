import collections

import itertools
from random import shuffle
import numpy

from data_loader import InputFeatures, InputExample


# Note that the order of the keys matters. (alphabetic order taken)
SNIPS_CLASS_2_RATIONALE_V1 = collections.OrderedDict()
SNIPS_CLASS_2_RATIONALE_V1['AddToPlaylist'] = ['add', 'playlist', 'album', 'list']
SNIPS_CLASS_2_RATIONALE_V1['BookRestaurant'] = ['book', 'restaurant', 'reservation', 'reservations']
SNIPS_CLASS_2_RATIONALE_V1['GetWeather'] = ['weather', 'forecast', 'warm', 'freezing', 'hot', 'cold']
SNIPS_CLASS_2_RATIONALE_V1['PlayMusic'] = ['play', 'music', 'song', 'hear']
SNIPS_CLASS_2_RATIONALE_V1['RateBook'] = ['rate', 'give', 'star', 'stars', 'points', 'rating', 'book']
SNIPS_CLASS_2_RATIONALE_V1['SearchCreativeWork'] = ['find', 'show', 'called']
SNIPS_CLASS_2_RATIONALE_V1['SearchScreeningEvent'] = ['movie', 'movies', 'find', 'theatres', 'cinema', 'cinemas', 'film', 'films', 'show']

def find_sub_token_ids(list_a, list_b):
    """
    Aims to find the location of list_b inside list_a
    :param list_a:
    :param list_b:
    :return: a list of {0, 1} with same length of list_a
    """
    assert set(list_b).issubset(set(list_a))

    pointer_a = 0
    pointer_b = 0
    location_of_b_in_a = [0 for _ in range(len(list_a))]

    if len(list_b) == 0:
        return location_of_b_in_a

    while pointer_a <= (len(list_a) - len(list_b)):
        if list_a[pointer_a] != list_b[pointer_b]:
            pointer_a += 1
            pointer_b = 0
            continue

        matched_flag = 0
        while list_a[pointer_a] == list_b[pointer_b]:
            matched_flag = 1
            pointer_a += 1
            pointer_b += 1
            if pointer_b == len(list_b):
                location_of_b_in_a[pointer_a - pointer_b:pointer_a] = [1 for _ in range(
                    len(location_of_b_in_a[pointer_a - pointer_b:pointer_a]))]

            if pointer_a == len(list_a):
                return location_of_b_in_a
            if pointer_b == len(list_b):
                pointer_b = 0

        if matched_flag == 1:
            pointer_b = 0

    return location_of_b_in_a

def find_token_frequency(examples, tokenizer):
    VOCAB_SIZE = tokenizer.vocab_size
    NUM_TOKENS = 0

    frequency = [0.0 for _ in range(VOCAB_SIZE)]
    all_tokens_in_train = [e.words for e in examples]
    all_tokens_in_train = list(itertools.chain(*all_tokens_in_train))

    for token in all_tokens_in_train:
        input_ids = tokenizer(token)['input_ids'][1:-1]
        for i in input_ids:
            NUM_TOKENS += 1
            frequency[i] += 1
        
    frequency = (numpy.array(frequency) / NUM_TOKENS).tolist()

    return frequency

def find_anntation_id_in_tokenizer_vocab(examples, tokenizer):
    """
    sentence: str
    return: list of list
            inner list is of length == len(tokenizer.tokenize(sentence))
    """
    ret = []
    for example in examples:
        # print(example.intent_label-1)
        label = list(SNIPS_CLASS_2_RATIONALE_V1.keys())[example.intent_label-1]
        annotations = list(set(example.words). \
                            intersection( \
                            set(SNIPS_CLASS_2_RATIONALE_V1[label])))
        
        # No rationales in this example
        if len(annotations) == 0:
            ret.append(None)
            continue
        
        sentence_str = ' '.join(example.words)
        sentence_input_ids = tokenizer(sentence_str)['input_ids']

        annotation_indicator = numpy.zeros((len(sentence_input_ids))).astype(int)
        for a in annotations:
            rationale_input_ids = tokenizer(a)['input_ids'][1:-1]
            annotation_indicator += numpy.array(find_sub_token_ids(sentence_input_ids, rationale_input_ids)).astype(int)

        ret.append(annotation_indicator)

    return ret




def augment_examples_and_to_features(examples, max_seq_len, tokenizer, replace, 
                                    pad_token_label_id=-100,
                                    cls_token_segment_id=0,
                                    pad_token_segment_id=0,
                                    sequence_a_segment_id=0,
                                    mask_padding_with_zero=True):

    check_one(examples, tokenizer)

    assert replace in ['random', 'frequency', 'bert']
    if replace == 'none':
        return None

    aug_features = []

    annotation_ids = find_anntation_id_in_tokenizer_vocab(examples, tokenizer)
    assert len(annotation_ids) == len(examples), "Different length of annotation_ids and examples"


    for example_id, example in enumerate(examples):
        aug_each_sample = []

        input_ids_mask_rationale = tokenizer(' '.join(example.words))['input_ids'].copy()
        input_ids_mask_non_rationale = input_ids_mask_rationale.copy()

        if replace == 'mask':
            num_rationale = sum(annotation_ids[example_id])
            if num_rationale > 0:
                # Mask rationale
                for i in range(len(input_ids_mask_rationale)):
                    if annotation_ids[example_id][i] == 1:
                        input_ids_mask_rationale[i] = tokenizer.mask_token_id
                
                # Mask non-rationale
                mask_non_rationale_cnt = 0
                random_index = list(range(len(input_ids_mask_non_rationale)))
                shuffle(random_index)
                for i in random_index:
                    if annotation_ids[example_id][i] == 0:
                        input_ids_mask_non_rationale[i] = tokenizer.mask_token_id
                        mask_non_rationale_cnt += 1
                        if mask_non_rationale_cnt == num_rationale:
                            break
                
                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_mask_rationale,
                                    attention_mask=example.attention_mask,
                                    token_type_ids=example.token_type_ids,
                                    intent_label_id=example.intent_label_id,
                                    slot_labels_ids=example.slot_labels_ids
                            ))

                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_mask_non_rationale,
                                    attention_mask=example.attention_mask,
                                    token_type_ids=example.token_type_ids,
                                    intent_label_id=example.intent_label_id,
                                    slot_labels_ids=example.slot_labels_ids
                            ))

            else:
                aug_each_sample.extend([None, None])
                

        elif replace == 'random':
            pass

        elif replace == 'frequency':
            frequency = find_token_frequency(examples, tokenizer)
            pass

        elif replace == 'bert':
            pass

        aug_features.append(aug_each_sample)
    
    return aug_features 



def check_one(examples, tokenizer):
    # check whether tokenizer(sentence str) == tokenizer(word by word) and concatnate

    for example in examples:
        input_ids_1 = tokenizer(' '.join(example.words))['input_ids'][1:-1]

        input_ids_2 = []
        for word in example.words:
            tokens = tokenizer.tokenize(word)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids_2 += ids

        if input_ids_1 != input_ids_2:
            print(example.words)
            print(input_ids_1)
            print(input_ids_2)
            exit()