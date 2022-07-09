import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from .torchcrf_weigted import CRF
from .module import IntentClassifier, SlotClassifier

from transformers import BertTokenizer

class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.tokenizer_debug = BertTokenizer.from_pretrained('./bert-base-uncased')

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        
        # for i in range(input_ids.size(0)):
        #     print(self.tokenizer_debug.convert_ids_to_tokens(input_ids[i]))
        #     print(attention_mask[i])
        #     print(token_type_ids[i])
        
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        # if torch.isnan(pooled_output).any():
        #     print("pooled_output nan\n",pooled_output)
        #     print("outputs:", outputs)
        #     torch.save(self.bert.state_dict(), "./nan_model_state")
        #     print("NAN model saved!!")
        #     exit(-1)
        
        # if torch.isnan(sequence_output).any():
        #     print("sequence_output nan\n",sequence_output)
        #     exit(-1)
        

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += (1-self.args.slot_loss_coef) * intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                # if torch.isnan(slot_logits).any():
                #     print("slot_logits forward nan\n")
                #     print(slot_logits)
                #     exit(-1)
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


    def forward_aug(self, batch_aug:list, aug_flags):
        # print("Forward aug")
        # print(aug_flags)
        # print(batch_aug[0].shape)
        # print(batch_aug[1].shape)
        # print(batch_aug[2].shape)
        # print(batch_aug[3].shape)
        # print(batch_aug[4].shape)
        
        if sum(aug_flags)==0:
            print("Batch without any annotated sample!")
            # exit(-1)
            return 0

        indices_with_annotation = self.find_1s_index(aug_flags).to(self.device)
        num_sample_with_rationale = len(indices_with_annotation)

        num_aug_each_sample = 1 if self.args.replace == 'mask' else self.args.replace_repeat
        bz = num_sample_with_rationale*num_aug_each_sample*2

        # print("input_ids shape:", batch_aug[0].shape)

        input_ids = torch.index_select(batch_aug[0], 0, indices_with_annotation).view(bz, -1)
        attention_mask = torch.index_select(batch_aug[1], 0, indices_with_annotation).view(bz, -1)
        token_type_ids = torch.index_select(batch_aug[2], 0, indices_with_annotation).view(bz, -1)
        intent_label_ids = torch.index_select(batch_aug[3], 0, indices_with_annotation).view(bz, -1)
        slot_label_ids = torch.index_select(batch_aug[4], 0, indices_with_annotation).view(bz, -1)
        weight = torch.index_select(batch_aug[5], 0, indices_with_annotation)

        # print("input_ids shape:", input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        # print(intent_label_ids.shape)
        # print(slot_label_ids.shape)
        # print(weight.shape)
        # exit()

        # temp_tokenizer = self.load_tokenizer()
        # for i in range(input_ids.size(0)):
        #     for j in range(2*self.args.replace_repeat):
        #         words = temp_tokenizer.decode(input_ids[i][j])
        #         print(words)
        # exit()

        # tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        # for i in range(input_ids.size(0)):
        #     print(tokenizer.convert_ids_to_tokens(input_ids[i]))
        #     print(attention_mask[i])
        #     print(token_type_ids[i])
        # print()
        # exit()

        print("Aug data forward...")

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.args.sub_task == 'intent':
            intent_logits = self.intent_classifier(pooled_output)
            intent_likelihood = torch.softmax(intent_logits, dim=1)

            intent_likelihood_on_gold = torch.gather(intent_likelihood, 1, intent_label_ids)
            intent_likelihood_on_gold = intent_likelihood_on_gold.view(num_sample_with_rationale, num_aug_each_sample*2)
            weighted_likelihood_on_gold = intent_likelihood_on_gold * weight

            likelihood_rationale_replaced = weighted_likelihood_on_gold[:, :num_aug_each_sample]
            likelihood_non_rationale_replaced = weighted_likelihood_on_gold[:, num_aug_each_sample:]

            m_1 = self.log_odds(likelihood_rationale_replaced.sum(dim=1, keepdim=True)) # Refer to m in "...input marginalization"
            m_2 = self.log_odds(likelihood_non_rationale_replaced.sum(dim=1, keepdim=True))

            margin_loss = self.margin_loss(m_2, m_1, self.args.margin)

            # print(margin_loss.shape)
            return margin_loss

        elif self.args.sub_task == 'slot':
            slot_logits = self.slot_classifier(sequence_output) # snips 74 label dimension
            # print("===========")
            # print(sequence_output.shape)
            # print(slot_logits.shape)
            # print(slot_logits)
            
            if slot_label_ids is not None:
                if self.args.use_crf:

                    # if torch.isnan(slot_logits).any():
                    #     print("Slot logits nan:", slot_logits)
                    #     exit(-1)
                    # print("slot_logits \n", slot_logits)
                    likelihood_rationale_replaced, likelihood_non_rationale_replaced = self.crf(slot_logits, slot_label_ids, mask=attention_mask.byte(), reduction='weighted', weight=weight, aug_num_each_sample=num_aug_each_sample)

                    # print("likelihood_rationale_replaced", likelihood_rationale_replaced)
                    # print()
                    # print("likelihood_non_rationale_replaced", likelihood_non_rationale_replaced)

                    m_1 = self.log_odds(likelihood_rationale_replaced.sum(dim=1, keepdim=True))
                    m_2 = self.log_odds(likelihood_non_rationale_replaced.sum(dim=1, keepdim=True))

                    # print("m_1:", m_1)
                    # print("m_2:", m_2)

                    margin_loss = self.margin_loss(m_2, m_1, self.args.margin)
                    # print(margin_loss.shape)
                    # exit()
                    return margin_loss



    @staticmethod
    def find_1s_index(t):
        ret = []
        assert len(t.size())==1

        for i, _item in enumerate(t.tolist()):
            if _item == 1:
                ret.append(i)
        return torch.tensor(ret)
    
    @staticmethod
    def load_tokenizer():
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained('./bert-base-uncased')

    @staticmethod
    def log_odds(x):
        ret = torch.zeros(x.size()).float().fill_(1.).cuda()

        # if (x<=0).any() or (x>=1).any():
            # print(x)
            # print("Error log odds")
            # exit(-1)
            # x[x==0] += 1e-50
        
        # odds = x/(1-x)
        pos_valid = (x!=0)
        ret[pos_valid] = (x/(1-x))[pos_valid]
        return ret.log2()


    @staticmethod
    def margin_loss(x, y, epsilon):
        """
        Expect that X is EPSILON larger than Y
        :param x: Expected to be larger than y
        :param y: Expected to be smaller than x
        :param epsilon: margin
        :return: Marginal loss
        """
        assert epsilon > 0
        epsilon = torch.full(x.size(), epsilon).cuda()
        return torch.mean(torch.max(torch.tensor(0.0).cuda(), y-x+epsilon))
