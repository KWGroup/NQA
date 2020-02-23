from tensorflow import keras 
import tensorflow as tf
from transformers import TFAlbertPreTrainedModel, TFAlbertModel, AlbertConfig
from transformers.modeling_tf_utils import get_initializer
###################### I: Data Warping ##########################################
# Functions below are for creating the keras models.
#  
# * TFAlbertForSequenceClassification is for predicting the class label of 
# the yesno short answer and whether a candidate contain_answer. 
# 
# * TFAlbertForQuestionAnswering is for predicting the start and end 
# token of the short answer. 
# 
# flags: 
##  
SHORT_ANS_YESNO = 'short_ans_yesno'
SHORT_ANS_ENTITY = 'short_ans_entity'
CANDIDATE_FILTER = 'candidate_filter'
SHORT_ANS = 'short_answer'
## 
#################################################################################
class TFAlbertForSequenceClassification(TFAlbertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    super(TFAlbertForSequenceClassification, self).__init__(config, *inputs, **kwargs)
    self.num_labels = config.num_labels

    self.albert = TFAlbertModel(config, name="albert")
    self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
    self.classifier = tf.keras.layers.Dense(
      config.num_labels, kernel_initializer=get_initializer(config.initializer_range), 
      name="classifier"
    )

  def call(self, inputs, **kwargs):
    outputs = self.albert(inputs, **kwargs)

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))
    logits = self.classifier(pooled_output)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
    return outputs  # logits, (hidden_states), (attentions)
class TFAlbertForQuestionAnswering(TFAlbertPreTrainedModel):
  def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.num_labels = config.num_labels

    self.albert = TFAlbertModel(config, name="albert")
    self.qa_outputs = tf.keras.layers.Dense(
      config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
    )

  def call(self, inputs, **kwargs):  
    outputs = self.albert(inputs, **kwargs)

    sequence_output = outputs[0]

    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)

    outputs = (start_logits, end_logits,) + outputs[2:]

    return outputs  # start_logits, end_logits, (hidden_states), (attentions)

def create_model(task=SHORT_ANS_YESNO):
  assert task in {SHORT_ANS_YESNO, SHORT_ANS_ENTITY,CANDIDATE_FILTER}, \
    f"task should be {SHORT_ANS_YESNO}, {SHORT_ANS_ENTITY}, or {CANDIDATE_FILTER}"

  # input layers
  token_ids = keras.Input(shape=(512,), dtype='int32', name='token_ids')
  segment_ids = keras.Input(shape=(512,), dtype='int32', name='segment_ids')
  mask_ids = keras.Input(shape=(512,), dtype='int32', name='mask_ids')

  if task == SHORT_ANS_YESNO or task == CANDIDATE_FILTER:
    if task == SHORT_ANS_YESNO:
      config = AlbertConfig.from_pretrained('albert-base-v2', num_labels=3)
    else: # candidate filter 
      config = AlbertConfig.from_pretrained('albert-base-v2', num_labels=2)
    albert_qa_layer = TFAlbertForSequenceClassification(config)
  else:
    albert_qa_layer = TFAlbertForQuestionAnswering.from_pretrained('albert-base-v2')

  # both tasks use the same input format
  albert_qa_outputs = albert_qa_layer([token_ids, mask_ids, segment_ids])

  if task == SHORT_ANS_YESNO or task == CANDIDATE_FILTER:
    logits = albert_qa_outputs[0]

    # create model
    model = keras.Model(
      inputs=[token_ids, mask_ids, segment_ids], 
      outputs=[logits]
    )
  else:
    start_logits, end_logits = albert_qa_outputs[:2]

    # create model
    model = keras.Model(
      inputs=[token_ids, mask_ids, segment_ids], 
      outputs=[start_logits, end_logits]
    )

  return model