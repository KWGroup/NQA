import zipfile 
import numpy as np
import json
import pandas as pd
###################### I: Data Warping ##########################################
# Functions below are for warping the training data. They are design for 
# handling three tasks: 
# 1. "short_ans_entity" in short answer identification 
# 2. "short_ans_yesno" (YES/NO/None) of the short answer identification 
# 3. "candidate_filter"  : determine whether a candidate contains the answer 
# 4. "global_identifier" : identifiying the long answer from all candidate 
#
#################################################################################
# TODO: 
# 1. [X] Allow different output modes: 1. pd.dataframe, 2. iterrows generator 
# 2. [X] Warping differently for candidate filtering and global_identifier 
# *  No need for dependency matrix for candidate filter.  
# *  But it needs to have two modes:  
#    - mode 1: Assigning long answer to all parent candidates and use all 
#              candidate for training 
#    - mode 2: Remove the parent candidates and use the candidate of the lowest 
#              level only. 
#              this way, we can have less amount of candidate. 
# *  [X] For global identifier, we need all candidate and the dependency matrix. 
#################################################################################
def get_train_data():
  json_zip_name = 'simplified-nq-train.jsonl.zip'
  with zipfile.ZipFile(json_zip_name) as myzip:
    with myzip.open('simplified-nq-train.jsonl') as my_json_file:
      for json_object_str in my_json_file:
        yield  json.loads(json_object_str.decode("utf-8"))

# Preprocessing function for all tasks 
def extracting_text_using_start_end_token_id(document_text, start_token, end_token):
    splitted_document_text = document_text.split()
    return ' '.join(splitted_document_text[start_token:end_token])

# These are the functions with cure's YES/NO,NULL classification 
# fused with dexter's short answer labels 
def has_long_answer(long_answer_candidate):
  return long_answer_candidate['start_token'] != -1 \
  and long_answer_candidate['candidate_index'] != -1 \
  and long_answer_candidate['end_token'] != -1

def data_cleaning_for_short_answer(
  json_obj,
  task='both',
  example_id=False):
  '''
  keys of the output dictionary: 
    'example_id' # optional 
    'question_text'
    'long_answer_text'
    'yes_no_answer' # exist only if task == 'both' or 'short_ans_yesno'
    'short_answer_start_token' # exist only  if task == 'both' or 'short_ans_entity'
    'short_answer_end_token' # exist only if task == 'both' or 'short_ans_entity'
  ''' 
  assert task == 'short_ans_yesno' or task == 'short_ans_entity' or task == 'both'
  new_data_d = {}
  # assignment for both tasks  
  annotations = json_obj['annotations'][0]
  long_answer_candidate = annotations['long_answer']
  if example_id:
    new_data_d['example_id'] = json_obj['example_id']
  new_data_d['question_text'] = json_obj['question_text']
  long_ans_start = long_answer_candidate['start_token']
  long_ans_end = long_answer_candidate['end_token']
  new_data_d['long_answer_text'] = \
    extracting_text_using_start_end_token_id(
            json_obj['document_text'],
            long_ans_start,
            long_ans_end
        )
  if task != 'both':
    if task == 'short_ans_entity':
      short_answer_candidate = annotations['short_answers']
      if not short_answer_candidate:
        short_ans_start = -1
        short_ans_end = -1
      else:
        short_ans_start = short_answer_candidate[0]['start_token'] - long_ans_start
        short_ans_end = short_answer_candidate[0]['end_token'] - long_ans_start
      new_data_d['short_answer_start_token'] = short_ans_start
      new_data_d['short_answer_end_token'] = short_ans_end
    elif task == 'short_ans_yesno':
      new_data_d['yes_no_answer'] = annotations['yes_no_answer']
  else:
    # get short_ans_entity labels 
    short_answer_candidate = annotations['short_answers']
    if not short_answer_candidate:
      short_ans_start = -1
      short_ans_end = -1
    else:
      short_ans_start = short_answer_candidate[0]['start_token'] - long_ans_start
      short_ans_end = short_answer_candidate[0]['end_token'] - long_ans_start
    new_data_d['short_answer_start_token'] = short_ans_start
    new_data_d['short_answer_end_token'] = short_ans_end
    # get short_ans_yesno labels 
    new_data_d['yes_no_answer'] = annotations['yes_no_answer']
  return new_data_d 

'''def data_cleaning_for_long_answer(old_data_d):
  # keys of the output dict 
  # 1. question_text 
  # 2. candidate_text_list 
  # 3. answering_candidate_id : candidate id of the long answer 
  # 4. dependency_matrix 
  new_data_d = {} 
  # splitting the document text
  document_text = old_data_d['document_text'].split()
  # assigning question text   
  new_data_d['question_text'] = old_data_d['question_text']
  # assigning candidate text 
  candidates = old_data_d['long_answer_candidates']
  document_text = old_data_d['document_text'] 
  candidate_text_list = []
  for a_candidate in candidates:
    start_token = a_candidate['start_token']
    end_token = a_candidate['end_token']
    candidate_text = extracting_text_using_start_end_token_id(document_text, start_token, end_token)
    candidate_text_list.append(candidate_text)
  new_data_d['candidate_text_list'] = candidate_text_list
  # assigning answering ids 
  new_data_d['answering_candidate_id'] = old_data_d['annotations'][0]['long_answer']['candidate_index']
  def get_candidate_dependency_matrix(candidates):
    def dependency_determinator(candidate_A,candidate_B):
      # Input two candidates: A and B 
      # Output whether A belong to B 
      stA_ = candidate_A['start_token']
      etA_ = candidate_A['end_token']
      stB_ = candidate_B['start_token']
      etB_ = candidate_B['end_token']
      if stB_<=stA_ and etA_<=etB_:
        return True    # A belong to B
      else:
        return False   # A not belong to B 
    dependency_results = [[dependency_determinator(candidate_A,candidate_B) for candidate_A in candidates] for candidate_B in candidates] 
    dependency_matrix = np.array(dependency_results)
    # remove diagonal results
    for i in range(dependency_matrix.shape[0]):
      dependency_matrix[i][i]=None 
    return dependency_matrix
  # assigning dependency matrix 
  new_data_d['dependency_matrix'] = get_candidate_dependency_matrix(candidates)
  return new_data_d '''

def data_cleaning_for_long_answer(old_data_d, task='local', mode = 'reduce_candidate'): 
  assert task == 'local' or task == 'global' 
  assert mode == 'extend_answer' or mode == 'reduce_candidate'
  # Input: 
  # Task: global or local (i.e., candidate filtering)
  # When task == local, there are two modes:  
  # 1. extend_answer: extend the answer candidate to the parents of 
  #    the original answer candidate.  
  #    long answer result. 
  # 2. reduce_candidate: remove all child and parent of the answer candidate 
  # Output: 
  # Keys of the output dict 
  # 1. question_text 
  # 2. candidate_text_list 
  # 3. answering_candidate_id : candidate id of the long answer 
  # 4. dependency_matrix 
  new_data_d = {} 
  # splitting the document text
  document_text = old_data_d['document_text'].split()
  # assigning question text   
  new_data_d['question_text'] = old_data_d['question_text']
  # assigning candidate text 
  candidates = old_data_d['long_answer_candidates']
  document_text = old_data_d['document_text'] 
  candidate_text_list = []
  for a_candidate in candidates:
    start_token = a_candidate['start_token']
    end_token = a_candidate['end_token']
    candidate_text = extracting_text_using_start_end_token_id(document_text, start_token, end_token)
    candidate_text_list.append(candidate_text)
  new_data_d['candidate_text_list'] = candidate_text_list
  # assigning answering ids 
  new_data_d['answering_candidate_id'] = old_data_d['annotations'][0]['long_answer']['candidate_index']
  def get_candidate_dependency_matrix(candidates):
    def dependency_determinator(candidate_A,candidate_B):
      # Input two candidates: A and B 
      # Output whether A belong to B 
      stA_ = candidate_A['start_token']
      etA_ = candidate_A['end_token']
      stB_ = candidate_B['start_token']
      etB_ = candidate_B['end_token']
      if stB_<=stA_ and etA_<=etB_:
        return True    # A belong to B
      else:
        return False   # A not belong to B 
    dependency_results = [[dependency_determinator(candidate_A,candidate_B) for candidate_A in candidates] for candidate_B in candidates] 
    dependency_matrix = np.array(dependency_results)
    # remove diagonal results
    for i in range(dependency_matrix.shape[0]):
      dependency_matrix[i][i]=None 
    return dependency_matrix
  # assigning dependency matrix 
  dependency_matrix = get_candidate_dependency_matrix(candidates) 
  if task == 'global':
    new_data_d['dependency_matrix'] = dependency_matrix 
  else: # task == 'local': 
    assert new_data_d['answering_candidate_id']!=-1
    def get_parent_ids(dependency_matrix,target_candidate_id):
      return np.where(dependency_matrix[:,target_candidate_id])[0].tolist()
    def get_child_ids(dependency_matrix,target_candidate_id):
      return np.where(dependency_matrix[target_candidate_id,:])[0].tolist()
    if mode == 'extend_answer':
      extended_answer_ids = get_parent_ids(dependency_matrix,new_data_d['answering_candidate_id']) 
      new_data_d['answering_candidate_id'] = [new_data_d['answering_candidate_id']]+extended_answer_ids 
    else: # mode == 'reduce_candidate' 
      reduce_ids = get_parent_ids(dependency_matrix,new_data_d['answering_candidate_id'])
      reduce_ids += get_child_ids(dependency_matrix,new_data_d['answering_candidate_id'])
      # revise the answering_candidate_id and candidate_text_list
      answer_candidate_text = candidate_text_list[new_data_d['answering_candidate_id']]
      new_candidate_text_list = [text for i, text in enumerate(candidate_text_list) if i not in reduce_ids] 
      new_data_d['candidate_text_list'] = new_candidate_text_list 
      new_data_d['answering_candidate_id'] = new_candidate_text_list.index(answer_candidate_text) 
  return new_data_d

def create_answer_data_generator(
  input,
  task='short_ans_entity',
  candidate_filtering_mode = 'reduce_candidate'):
  if type(input) == str: 
    f = open(input,'r')
    input_generator = map(json.loads,f) 
  else:
    input_generator = input
  assert candidate_filtering_mode == 'extend_answer' or candidate_filtering_mode == 'reduce_candidate'
  assert task == 'short_ans_entity' or task == 'short_ans_yesno' or task == 'short_answer' \
    or task == 'candidate_filter' or task == 'global_identification' or task == 'long_answer' 
  answer_dataset = []
  for old_data_d in input_generator:
    if task != 'global_identification' and task != 'long_answer':
      # when global identification is not enabled, the examples without long answer 
      # should be removed. 
      if has_long_answer(old_data_d['annotations'][0]['long_answer']):
        if task == 'short_ans_entity' or task == 'short_ans_yesno' or task == 'short_answer':
          if task == 'short_answer':
            new_data_d = data_cleaning_for_short_answer(old_data_d,task = 'both')
          else:
            new_data_d = data_cleaning_for_short_answer(old_data_d,task = task) 
        else: # long answer case 
          new_data_d = data_cleaning_for_long_answer(old_data_d,task = 'local', mode = candidate_filtering_mode)
        yield new_data_d  
    else: # task == 'global_identification' or 'long_answer' 
      new_data_d = data_cleaning_for_long_answer(old_data_d, task = 'global')
      yield new_data_d  
def data_warping_for_candidate_filter(raw_generator):
  for example in raw_generator:
    question_text = example['question_text'] 
    answering_candidate_id = example['answering_candidate_id']
    for i, candidate_text in enumerate(example['candidate_text_list']): 
      if type(answering_candidate_id) == list: 
        label = i in answering_candidate_id
      else:
        label = i == answering_candidate_id 
      yield {
          'question_text':question_text, 
          'candidate_text':candidate_text, 
          'contain_answer':label
        }
def create_answer_dataset(
  input,
  task='short_ans_entity',
  candidate_filtering_mode = 'reduce_candidate'
  ):
  assert candidate_filtering_mode == 'extend_answer' or candidate_filtering_mode == 'reduce_candidate'
  assert task == 'short_ans_entity' or task == 'short_ans_yesno' or task == 'short_answer' \
    or task == 'candidate_filter' or task == 'global_identification' or task == 'long_answer' 
  answer_dataset = []
  data_generator = create_answer_data_generator(
    input,
    task=task,
    candidate_filtering_mode = candidate_filtering_mode)
  if task == 'candidate_filter':
    data_generator = data_warping_for_candidate_filter(data_generator)
  for new_data_d in data_generator: 
    answer_dataset.append(new_data_d)
  return pd.DataFrame(answer_dataset)
###################### II: Data Formatting #############################################
# Functions below are for generating instances that match with
# the format of training input and outuput. 
#######################################################################################
def get_question_tokens(tokenizer, question_text):
  question_tokens = ['[CLS]'] + tokenizer.tokenize(question_text) + ['[SEP]']
  return question_tokens
def get_question_wiki_text_tokens(tokenizer, question_tokens,wiki_text):
  tokens = question_tokens + tokenizer.tokenize(wiki_text)\
    + ['[SEP]']
  return tokens 
def get_question_long_answer_tokens_and_start_end_tokens(
  tokenizer, 
  question_tokens,
  long_answer_text,
  short_answer_start_token, 
  short_answer_end_token
  ):
  long_answer_tokens = long_answer_text.split()
  chunk_1 = ' '.join(long_answer_tokens[:short_answer_start_token])
  chunk_2 = ' '.join(long_answer_tokens[short_answer_start_token:short_answer_end_token])
  chunk_3 = ' '.join(long_answer_tokens[short_answer_end_token:])
  # handle new start end token
  tokens = question_tokens + tokenizer.tokenize(chunk_1)
  label_start_token = len(tokens)
  tokens = tokens + tokenizer.tokenize(chunk_2)
  label_end_token = len(tokens)
  tokens = tokens + tokenizer.tokenize(chunk_3) + ['[SEP]']
  return tokens, label_start_token, label_end_token
def generate_input_feature(
    tokenizer,
    question_text,
    wiki_text, 
    squad = False, 
    original_start = -1, 
    original_end = -1,
    MAX_LENGTH = 512
    ):
  question_tokens = get_question_tokens(
    tokenizer,
    question_text) 
  sentence_A_len = len(question_tokens)
  if original_start == -1 or squad == False:
    tokens = get_question_wiki_text_tokens(
      tokenizer, 
      question_tokens,
      wiki_text)
    if squad:
      label_start_token = 0
      label_end_token = 0
  else:
    tokens, label_start_token, label_end_token = get_question_long_answer_tokens_and_start_end_tokens(
      tokenizer, 
      question_tokens,
      wiki_text,
      original_start, 
      original_end
      )
  sentence_len = len(tokens)
  # apply truncating 
  if sentence_len > MAX_LENGTH:
    tokens = tokens[:MAX_LENGTH-1] + ['[SEP]']
    sentence_len = MAX_LENGTH
  if squad and label_end_token > MAX_LENGTH - 1: # should not exceed last token [SEP]
    label_start_token = 0
    label_end_token = 0
  # create segment_id and mask_id
  segment_ids = sentence_A_len * [0] + (sentence_len - sentence_A_len) * [1] 
  mask_ids = sentence_len * [1]
  # apply padding
  if (sentence_len < MAX_LENGTH):
    pad_len = MAX_LENGTH - sentence_len
    tokens = tokens + pad_len * ['[PAD]']
    segment_ids = segment_ids + pad_len * [0]
    mask_ids = mask_ids + pad_len * [0]
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  if squad:
    return (token_ids, segment_ids, mask_ids), (label_start_token, label_end_token) 
  else:
    return token_ids, segment_ids, mask_ids

class ItemMapper():
  def __init__(self):
    self.key_to_id_map = None
  def get(self,instance,key):
    if type(instance) == dict:
      return instance[key]
    else:
      if self.key_to_id_map == None:
        self.key_to_id_map = dict(map(lambda x:(x[1],x[0]),enumerate(instance.keys())))
      return instance[self.key_to_id_map[key]] 

def generate_input_output_per_row(row, task, tokenizer,instance_item_mapper):
  # This function takes a row of the short answer dataframe as input
  # and outputs a dict with the following 
  # keys: 
  # 1. token_ids
  # 2. segment_ids 
  # 3. mask_ids 
  # 4. label_yes_no (if task == 'short_ans_yesno' or 'short_answer')  
  # 5. label_start/end_token (if task == 'short_ans_yesno' or 'short_ans_entity') 
  # 6. label_contain_answer (if task == 'candidate_filter') 
  label_yes_no_map = {
    'YES': 0,
    'NO': 1,
    'NONE': 2
  }
  short_ans_feature_dict = {}
  if task == 'short_ans_entity':
    (token_ids, segment_ids, mask_ids), (label_start_token, label_end_token) = \
      generate_input_feature(
          tokenizer,
          instance_item_mapper(row,"question_text"),
          instance_item_mapper(row,"long_answer_text"), 
          squad = True, 
          original_start = instance_item_mapper(row,"short_answer_start_token"),
          original_end = instance_item_mapper(row,"short_answer_end_token")
          )
  elif task == 'short_ans_yesno':
    token_ids, segment_ids, mask_ids = \
      generate_input_feature(
          tokenizer,
          instance_item_mapper(row,"question_text"),
          instance_item_mapper(row,"long_answer_text"))
    short_ans_feature_dict['label_yes_no'] = label_yes_no_map[instance_item_mapper(row,"yes_no_answer")]
  elif task == 'short_answer':
    (token_ids, segment_ids, mask_ids), (label_start_token, label_end_token) = \
      generate_input_feature(
          tokenizer,
          instance_item_mapper(row,"question_text"),
          instance_item_mapper(row,"long_answer_text"), 
          squad = True, 
          original_start = instance_item_mapper(row,"short_answer_start_token"),
          original_end = instance_item_mapper(row,"short_answer_end_token")
          )
    short_ans_feature_dict['label_yes_no'] = label_yes_no_map[instance_item_mapper(row,"yes_no_answer")]
  elif task == 'candidate_filter':
    token_ids, segment_ids, mask_ids = \
      generate_input_feature(
          tokenizer,
          instance_item_mapper(row,"question_text"),
          instance_item_mapper(row,"candidate_text"))
    short_ans_feature_dict['label_contain_answer'] = int(instance_item_mapper(row,"contain_answer"))
  else: # task == 'global_identification' 
    pass 
  short_ans_feature_dict['token_ids'] = token_ids
  short_ans_feature_dict['segment_ids'] = segment_ids
  short_ans_feature_dict['mask_ids'] = mask_ids
  if task == 'short_ans_entity' or task == 'short_answer':
    short_ans_feature_dict['label_start_token'] = label_start_token
    short_ans_feature_dict['label_end_token'] = label_end_token
  return short_ans_feature_dict
def input_output_feature_generator(input, tokenizer, task = 'short_ans_yesno'):
  # assertions 
  instance_item_mapper = ItemMapper().get
  assert task == 'short_ans_yesno' or task == 'short_ans_entity' or task == 'short_answer' or task == 'candidate_filter' or task == 'global_identification'
  if type(input).__name__ == 'generator':
    for i, instance in enumerate(input):
      if i == 0: # do assertion in the beginning 
        if task == 'short_ans_yesno':
          assert "yes_no_answer" in instance.keys() 
          assert "long_answer_text" in instance.keys()
        elif task == 'short_ans_entity':
          assert "short_answer_start_token" in instance.keys()
          assert "short_answer_end_token" in instance.keys()
          assert "long_answer_text" in instance.keys()
        elif task == 'short_answer':
          assert "yes_no_answer" in instance.keys()
          assert "short_answer_start_token" in instance.keys()
          assert "short_answer_end_token" in instance.keys()
          assert "long_answer_text" in instance.keys()
        elif task == 'candidate_filter':
          assert "contain_answer" in instance.keys()
          assert "candidate_text" in instance.keys()
      #_, row = next(pd.DataFrame([instance]).iterrows())
      yield generate_input_output_per_row(
          instance,
          task,
          tokenizer,
          instance_item_mapper)
  else: # input is a pandas dataframe 
    if task == 'short_ans_yesno':
      assert "yes_no_answer" in input.columns 
      assert "long_answer_text" in input.columns
    elif task == 'short_ans_entity':
      assert "short_answer_start_token" in input.columns 
      assert "short_answer_end_token" in input.columns
      assert "long_answer_text" in input.columns 
    elif task == 'short_answer':
      assert "yes_no_answer" in input.columns 
      assert "short_answer_start_token" in input.columns 
      assert "short_answer_end_token" in input.columns 
      assert "long_answer_text" in input.columns
    elif task == 'candidate_filter':
      assert "contain_answer" in input.columns 
      assert "candidate_text" in input.columns
    for _, row in input.iterrows():
      yield generate_input_output_per_row(row,task,tokenizer,instance_item_mapper)
def create_input_output_featureset(
  input, 
  tokenizer, 
  task='short_ans_yesno'):
  '''
  parameters:
  raw_df: short answer dataframe
  task: 'short_ans_yesno' (default): yes/no answer; 'short_ans_entity': short answer entity
  returns:
  dataframe of tokenized short answer dataset
  '''
  generator = input_output_feature_generator(input, tokenizer, task = task)
  dict_list = [element for element in generator]
  return pd.DataFrame(dict_list)