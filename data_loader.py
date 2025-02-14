import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)

import collections
import itertools
import random
from random import sample, shuffle
import numpy
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

        self.annotation_flag = False

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids, weight=-1, augmented_flag=0):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids
        self.weight = weight # Used for augmented samples. Change later
        self.augmented_flag = augmented_flag

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


processors = {
    "atis": JointProcessor,
    "snips": JointProcessor
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids,
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    # if os.path.exists(cached_features_file):
    if False:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
            examples = few_shot(args, examples)
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)


        if mode == 'train' and args.replace != 'none':
            aug_features, aug_flags = augment_examples_and_conver_to_features(examples, args.max_seq_len, tokenizer, args.replace, args,
                                                pad_token_label_id=pad_token_label_id)
            logger.info("Saving features_aug into cached file %s", cached_features_file)

            # Modify augmented_flag in origin examples
            modify_aug_flag(features, aug_flags)
            # torch.save(aug_examples, cached_features_file+'-aug')

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    all_aug_flag = torch.tensor([f.augmented_flag for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids, all_aug_flag)

    if mode == 'train' and args.replace != 'none':
        # for l in aug_features:
        #     print(l[0].input_ids)
        #     exit()

        all_input_ids_aug = torch.tensor([[f.input_ids for f in _list] for _list in aug_features], dtype=torch.long)
        all_attention_mask_aug = torch.tensor([[f.attention_mask for f in _list] for _list in aug_features], dtype=torch.long)
        all_token_type_ids_aug = torch.tensor([[f.token_type_ids for f in _list] for _list in aug_features], dtype=torch.long)
        all_intent_label_ids_aug = torch.tensor([[f.intent_label_id for f in _list] for _list in aug_features], dtype=torch.long)
        all_slot_labels_ids_aug = torch.tensor([[f.slot_labels_ids for f in _list] for _list in aug_features], dtype=torch.long)
        all_weight_aug = torch.tensor([[f.weight for f in _list] for _list in aug_features], dtype=torch.float)


        # print(all_input_ids_aug.shape) # (numn_train, 2/args.repeat, max_seq_len)
        # print(all_attention_mask_aug.shape) # (numn_train, 2/args.repeat, max_seq_len)
        # print(all_token_type_ids_aug.shape) # (numn_train, 2/args.repeat, max_seq_len)
        # print(all_intent_label_ids_aug.shape) # (numn_train, 2/args.repeat)
        # print(all_slot_labels_ids_aug.shape) # (numn_train, 2/repeat, max_seq_len)
        # print(all_aug_flag_aug.shape)

        dataset_aug = TensorDataset(all_input_ids_aug, all_attention_mask_aug,
                                all_token_type_ids_aug, all_intent_label_ids_aug, all_slot_labels_ids_aug, all_weight_aug)
        return dataset, dataset_aug

    else:
        return dataset, None



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
    # assert set(list_b).issubset(set(list_a))

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
        # if len(annotations) == 0:
        #     ret.append([])
        #     continue
        
        sentence_str = ' '.join(example.words)
        sentence_input_ids = tokenizer(sentence_str)['input_ids']

        annotation_indicator = numpy.zeros((len(sentence_input_ids))).astype(int)
        for a in annotations:
            rationale_input_ids = tokenizer(a)['input_ids'][1:-1]
            annotation_indicator += numpy.array(find_sub_token_ids(sentence_input_ids, rationale_input_ids)).astype(int)

        ret.append(annotation_indicator.tolist())

    return ret




def augment_examples_and_conver_to_features(examples, max_seq_len, tokenizer, replace, args,
                                    pad_token_label_id=-100,
                                    cls_token_segment_id=0,
                                    pad_token_segment_id=0,
                                    sequence_a_segment_id=0,
                                    mask_padding_with_zero=True):

    check_one(examples, tokenizer)

    pad_token_id = tokenizer.pad_token_id
    VOCAB_SIZE = tokenizer.vocab_size
    max_seq_cnt = -1

    assert replace in ['mask', 'random', 'frequency', 'bert']
    if replace == 'none':
        return None

    aug_features = []
    aug_flags = [0 for _ in range(len(examples))]

    annotation_ids = find_anntation_id_in_tokenizer_vocab(examples, tokenizer)
    assert len(annotation_ids) == len(examples), "Different length of annotation_ids and examples"

    bert_decider = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir='./bert-base-uncased/') if replace == 'bert' else None
    bert_decider = bert_decider.cuda() if replace == 'bert' else None
    frequency = find_token_frequency(examples, tokenizer) if replace == 'frequency' else None

    pbar = tqdm(examples)
    print("Augmenting data by {}".format(args.replace))
    for example_id, example in enumerate(pbar):
        
        slot_label_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            slot_label_ids.extend([int(slot_label)] + [pad_token_label_id]*(len(word_tokens)-1))

        attention_mask = [1. for _ in range(len(slot_label_ids)+2)]

        if len(attention_mask) > max_seq_cnt:
            max_seq_cnt = len(attention_mask)

        slot_label_ids = [pad_token_label_id] + slot_label_ids
        slot_label_ids += [pad_token_label_id] * (max_seq_len-len(slot_label_ids))
        attention_mask += [0. for _ in range(max_seq_len-len(attention_mask))]


        if len(attention_mask) > 50:
            print(len(attention_mask))
            print(example.words)

        token_type_id = torch.zeros(max_seq_len).float().tolist()
        intent_label_id = int(example.intent_label)

        # Complete input_ids
        origin_inputs = tokenizer(' '.join(example.words), return_tensors='pt')
        id_list = origin_inputs['input_ids'].tolist()[0]
        sentence_len = len(id_list)

        # print(id_list)
        # print(len(annotation_ids[example_id]))
        assert sentence_len == len(annotation_ids[example_id]), "Different length of sentence_len and annotation_len"

        aug_each_sample = []
        
        if replace == 'mask':
            input_ids_mask_rationale = tokenizer(' '.join(example.words))['input_ids'].copy()
            input_ids_mask_non_rationale = input_ids_mask_rationale.copy()

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
                
                input_ids_mask_rationale += [pad_token_id]*(max_seq_len-len(input_ids_mask_rationale))
                input_ids_mask_non_rationale += [pad_token_id]*(max_seq_len-len(input_ids_mask_non_rationale))

                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_mask_rationale,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_id,
                                    intent_label_id=intent_label_id,
                                    slot_labels_ids=slot_label_ids,
                                    augmented_flag=1,
                                    weight=1.,
                            ))

                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_mask_non_rationale,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_id,
                                    intent_label_id=intent_label_id,
                                    slot_labels_ids=slot_label_ids,
                                    augmented_flag=1,
                                    weight=1.,
                            ))
                
                aug_flags[example_id] = 1

            else:
                input_ids_empty = [-1 for _ in range(max_seq_len)] # Just empty to make dataset aligned, will not be used
                attention_mask_empty = input_ids_empty
                token_type_id_empty = input_ids_empty
                intent_label_id_empty = -1
                slot_label_ids_empty = input_ids_empty

                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_empty,
                                    attention_mask=attention_mask_empty,
                                    token_type_ids=token_type_id_empty,
                                    intent_label_id=intent_label_id_empty,
                                    slot_labels_ids=slot_label_ids_empty
                            ))
                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_empty,
                                    attention_mask=attention_mask_empty,
                                    token_type_ids=token_type_id_empty,
                                    intent_label_id=intent_label_id_empty,
                                    slot_labels_ids=slot_label_ids_empty
                            ))

                

                
        else:
            aug_each_sample = replace_rationale_all(args, tokenizer, 
                                                    sentence_len, annotation_ids, example_id, origin_inputs,
                                                    attention_mask, token_type_id, intent_label_id, slot_label_ids,
                                                    frequency, bert_decider, aug_flags
                                                    )

            aug_each_sample += replace_non_rationale_all(args, tokenizer, 
                                                         sentence_len, annotation_ids, example_id, origin_inputs,
                                                         attention_mask, token_type_id, intent_label_id, slot_label_ids,
                                                         frequency, bert_decider, aug_flags
                                                        )

        aug_features.append(aug_each_sample)


    # for i in range(4):
    #     aug_1, aug_2 = aug_features[i]
    #     if aug_1 is not None:
    #         print(tokenizer.tokenize(' '.join(examples[i].words)))
    #         print(examples[i].slot_labels)
    #         print(aug_1.input_ids)
    #         print(aug_1.attention_mask)
    #         print(aug_1.token_type_ids)
    #         print(aug_1.intent_label_id)
    #         print(aug_1.slot_labels_ids)
    #         print()
    #         exit()

    # print("Max seq len cnt: ", max_seq_cnt)
    # exit()
    
    return aug_features, aug_flags


def _sample(frequency, repeat):

        """
        return K samples based on input probability
        :param frequency: list of probabilities, sum(FREQUENCY) = 1
        :param k: repeat sampling w.r.t. FREQUENCY K times
        :return: the sampled index based on probability from FREQUENCY
        """

        token_indices_ret = []
        token_frequency_ret = []
        for repeat in range(repeat):
            x = random.uniform(0, 1)
            cumulative_probability = 0.0
            for token_index, item_probability in enumerate(frequency):
                cumulative_probability += item_probability
                if x < cumulative_probability:
                    token_indices_ret.append(token_index)
                    token_frequency_ret.append(item_probability)
                    break

        return torch.tensor(token_indices_ret), torch.tensor(token_frequency_ret) / sum(token_frequency_ret)

def modify_aug_flag(features, aug_flags):
    for i in range(len(aug_flags)):
        features[i].augmented_flag = aug_flags[i]

def replace_rationale_all(args, tokenizer, sentence_len, annotation_ids, example_id, origin_inputs, attention_mask, token_type_id, intent_label_id, slot_label_ids, frequency, bert_decider, aug_flags):
    aug_each_sample = []

    num_rationale = sum(annotation_ids[example_id])
    if num_rationale == 0:
        input_ids_empty = [-1 for _ in range(args.max_seq_len)]
        attention_mask_empty = input_ids_empty
        token_type_id_empty = input_ids_empty
        intent_label_id_empty = -1
        slot_label_ids_empty = input_ids_empty

        for r in range(args.replace_repeat):
            aug_each_sample.append(
                InputFeatures(input_ids=input_ids_empty,
                        attention_mask=attention_mask_empty,
                        token_type_ids=token_type_id_empty,
                        intent_label_id=intent_label_id_empty,
                        slot_labels_ids=slot_label_ids_empty,
                        weight=-1
                ))
        return aug_each_sample
            

    replace = args.replace
    repeat = args.replace_repeat
    VOCAB_SIZE = tokenizer.vocab_size

    j_th_token_save = []
    replace_candidate_index = []
    replace_candidate_likelihood = []

    vocab_index_list = list(range(VOCAB_SIZE))

    for j_th_token in range(sentence_len):
        if (j_th_token == 0) or (j_th_token == sentence_len - 1):
            continue
        if annotation_ids[example_id][j_th_token] == 0:
            continue

        j_th_token_save.append(j_th_token)

        if replace == 'random':
            replace_token_indices = random.sample(vocab_index_list, repeat)
            replace_token_likelihood = torch.full((len(replace_token_indices),), 1./repeat)
            
        elif replace == 'frequency':
            # frequency = find_token_frequency(examples, tokenizer)
            replace_token_indices, replace_token_likelihood = _sample(frequency, repeat)

        elif replace == 'bert':
            masked_inputs = copy.deepcopy(origin_inputs)
            masked_inputs = {key_: masked_inputs[key_].cuda() for key_ in masked_inputs}
            masked_inputs['input_ids'][0][j_th_token] = tokenizer.mask_token_id

            # bert_decider = BertForMaskedLM.from_pretrained('bert-base-uncsaed', cache_dir='./bert-base-uncased/')
            output = bert_decider(**masked_inputs)
            logits = output[0]

            topk_return = torch.topk(logits[0][j_th_token], repeat)
            replace_token_indices = topk_return.indices
            replace_token_logits = topk_return.values  # Used to calculate likelihood of replacing origin token with this token
            replace_token_likelihood = torch.softmax(replace_token_logits, dim=0)

        replace_candidate_index.append(replace_token_indices)
        replace_candidate_likelihood.append(replace_token_likelihood)

    renormalized_liklihood = sum(replace_candidate_likelihood) / len(replace_candidate_likelihood)

    for r in range(repeat):
        inputs_copy = copy.deepcopy(origin_inputs)
        mean_likelihood = renormalized_liklihood[r]
        for ith_rationale, j_th_token in enumerate(j_th_token_save):
            inputs_copy['input_ids'][0][j_th_token] = replace_candidate_index[ith_rationale][r]
        
        new_sentence = tokenizer.decode(inputs_copy['input_ids'][0][1:-1]) # debug
        
        if args.verbose:

            print("Replace rationale")
            print(new_sentence)
            print()
        
        input_ids = inputs_copy['input_ids'][0].tolist()
        input_ids += [0 for _ in range(args.max_seq_len-len(input_ids))]

        aug_each_sample.append(
            InputFeatures(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_id,
                        intent_label_id=intent_label_id,
                        slot_labels_ids=slot_label_ids,
                        weight=mean_likelihood,
                        augmented_flag=1
                ))

    aug_flags[example_id] = 1
    
    return aug_each_sample


def replace_non_rationale_all(args, tokenizer, sentence_len, annotation_ids, example_id, origin_inputs, attention_mask, token_type_id, intent_label_id, slot_label_ids, frequency, bert_decider, aug_flags):
    aug_each_sample = []

    num_rationale = sum(annotation_ids[example_id])
    if num_rationale == 0:
        input_ids_empty = [-1 for _ in range(args.max_seq_len)]
        attention_mask_empty = input_ids_empty
        token_type_id_empty = input_ids_empty
        intent_label_id_empty = -1
        slot_label_ids_empty = input_ids_empty

        for r in range(args.replace_repeat):
            aug_each_sample.append(
                InputFeatures(input_ids=input_ids_empty,
                        attention_mask=attention_mask_empty,
                        token_type_ids=token_type_id_empty,
                        intent_label_id=intent_label_id_empty,
                        slot_labels_ids=slot_label_ids_empty,
                        weight=-1
                ))
        return aug_each_sample

    replace = args.replace
    repeat = args.replace_repeat
    VOCAB_SIZE = tokenizer.vocab_size

    j_th_token_save = []
    replace_candidate_index = []
    replace_candidate_likelihood = []
    num_non_rationale_to_replace = num_rationale

    random_token_ids = list(range(sentence_len))
    shuffle(random_token_ids)

    for j_th_token in random_token_ids:
        if (j_th_token == 0) or (j_th_token == sentence_len - 1):
            continue
        if annotation_ids[example_id][j_th_token] == 1:
            continue

        j_th_token_save.append(j_th_token)
        num_non_rationale_to_replace -= 1

        if replace == 'random':
            vocab_index_list = list(range(VOCAB_SIZE))
            replace_token_indices = random.sample(vocab_index_list, repeat)
            replace_token_likelihood = torch.full((len(replace_token_indices),), 1./repeat)
            
        elif replace == 'frequency':
            # frequency = find_token_frequency(examples, tokenizer)
            replace_token_indices, replace_token_likelihood = _sample(frequency, repeat)

        elif replace == 'bert':
            masked_inputs = copy.deepcopy(origin_inputs)
            masked_inputs = {key_: masked_inputs[key_].cuda() for key_ in masked_inputs}
            masked_inputs['input_ids'][0][j_th_token] = tokenizer.mask_token_id
            # bert_decider = BertForMaskedLM.from_pretrained('bert-base-uncsaed', cache_dir='./bert-base-uncased/')
            output = bert_decider(**masked_inputs)
            logits = output[0]

            topk_return = torch.topk(logits[0][j_th_token], repeat)
            replace_token_indices = topk_return.indices
            replace_token_logits = topk_return.values  # Used to calculate likelihood of replacing origin token with this token
            replace_token_likelihood = torch.softmax(replace_token_logits, dim=0)

        replace_candidate_index.append(replace_token_indices)
        replace_candidate_likelihood.append(replace_token_likelihood)

        if num_non_rationale_to_replace == 0:
            break

    renormalized_liklihood = sum(replace_candidate_likelihood) / len(replace_candidate_likelihood)

    for r in range(repeat):
        inputs_copy = copy.deepcopy(origin_inputs)
        mean_likelihood = renormalized_liklihood[r]
        for ith_rationale, j_th_token in enumerate(j_th_token_save):
            inputs_copy['input_ids'][0][j_th_token] = replace_candidate_index[ith_rationale][r]
        
        new_sentence = tokenizer.decode(inputs_copy['input_ids'][0][1:-1]) # debug
        
        if args.verbose:
            print("Replace non-rationale:")
            print(new_sentence)
            print()
        
        input_ids = inputs_copy['input_ids'][0].tolist()
        input_ids += [0 for _ in range(args.max_seq_len-len(input_ids))]

        aug_each_sample.append(
            InputFeatures(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_id,
                        intent_label_id=intent_label_id,
                        slot_labels_ids=slot_label_ids,
                        weight=mean_likelihood,
                        augmented_flag=1
                ))
    aug_flags[example_id] = 1

    return aug_each_sample


def few_shot(args, examples):
    if args.shot == -1:
        return examples
    
    shuffle(examples)
    shot_each_class = dict()
    few_shot_examples = []
    for class_ in range(len(SNIPS_CLASS_2_RATIONALE_V1.keys())):
        shot_each_class[class_] = 0
    
    for example in examples:
        if shot_each_class[example.intent_label-1] < args.shot:
            few_shot_examples.append(example)
            shot_each_class[example.intent_label-1] += 1

            if sum(shot_each_class.values()) == (args.shot*len(SNIPS_CLASS_2_RATIONALE_V1.keys())):
                break
    
    return few_shot_examples


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