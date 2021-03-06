import numpy as np
import tensorflow as tf
import os 
def dataset_checker(dataset):
    '''
    This function fetch an instance from 
    a tf.data.Dataset object 
    '''
    if tf.__version__[0].split('.')[0] == '2':
        iterator = iter(dataset)
        return next(iterator)
    else:
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        with tf.Session() as sess:
            value = sess.run(one_element)
        return value


def df_to_dataset(dataframe,
                  batch_size,
                  task='short_ans_entity',
                  tpu_strategy=None
                  ):
    def formatting_dataframe(dataframe):
        dataframe_dict = dict(dataframe)
        new_dataframe_dict = dict()
        for key, value in dataframe_dict.items():
            new_value = np.vstack(dataframe_dict[key])
            new_dataframe_dict[key] = new_value
        return new_dataframe_dict

    dataframe = dataframe.copy()
    if task == 'short_ans_entity':
        label_start_token = dataframe.pop('label_start_token')
        label_end_token = dataframe.pop('label_end_token')
        dataset = tf.data.Dataset.from_tensor_slices(
            (formatting_dataframe(dataframe), label_start_token, label_end_token))
        dataset = dataset.map(lambda x,y1,y2:(x,(y1,y2)))
    elif task == 'short_ans_yesno':
        label_yes_no = dataframe.pop('label_yes_no')
        dataset = tf.data.Dataset.from_tensor_slices(
            (formatting_dataframe(dataframe), label_yes_no))
    elif task == 'short_answer':
        label_start_token = dataframe.pop('label_start_token')
        label_end_token = dataframe.pop('label_end_token')
        label_yes_no = dataframe.pop('label_yes_no')
        dataset = tf.data.Dataset.from_tensor_slices(
            (formatting_dataframe(dataframe), label_start_token, label_end_token, label_yes_no))
        dataset = dataset.map(lambda x,y1,y2,y3:(x,(y1,y2,y3))) 
    elif task == 'candidate_filter':
        label_contain_answer = dataframe.pop('label_contain_answer')
        dataset = tf.data.Dataset.from_tensor_slices(
            (formatting_dataframe(dataframe), label_contain_answer))
    dataset = (dataset
        .shuffle(buffer_size=len(dataframe))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        )

    return (
      tpu_strategy.experimental_distribute_dataset(dataset) 
      if tpu_strategy!=None else dataset
    )


def generator_to_dataset(input_generator,
                         batch_size,
                         task='candidate_filter',
                         shuffling_buffer_size=1000
                         ):
    def input_formatter(output_type):
        '''
        This function output information that
         guides the formating of input features. 

        Input description:
          `output_type` determine where to assign the formatting 
          information. 
          When output_type == 'data_convertor', it produces a function to 
            convert the data into the input feature space. 
          When output_type == 'tf_type', it produces a dict containing 
            the tf_type information of the input features. 
          When output_type == 'tf_shape', it produces a dict containing 
            the shape information of the input features. 
        '''
        if output_type == 'data_convertor':
            return lambda x: {'token_ids': x['token_ids'],
                              'segment_ids': x['segment_ids'],
                              'mask_ids': x['mask_ids']}
        elif output_type == 'tf_type':
            return {'token_ids': tf.int32,
                    'segment_ids': tf.int32,
                    'mask_ids': tf.int32}
        elif output_type == 'tf_shape':
            return {'token_ids': tf.TensorShape([512]),
                    'segment_ids': tf.TensorShape([512]),
                    'mask_ids': tf.TensorShape([512])}
    input_data_converter = input_formatter('data_convertor')
    input_tf_types = input_formatter('tf_type')
    input_tf_shapes = input_formatter('tf_shape')
    if task == 'candidate_filter':
        formatted_result_generator = map(
            lambda x: (input_data_converter(x),
                       x['label_contain_answer']),
            input_generator)
        label_count = 1
    elif task == 'short_answer':
        formatted_result_generator = map(
            lambda x: (input_data_converter(x),
                       (x['label_start_token'],
                       x['label_end_token'],
                       x['label_yes_no'])
                       ),
            input_generator)
        label_count = 3
    elif task == 'short_ans_entity':
        formatted_result_generator = map(
            lambda x: (input_data_converter(x),
                       (x['label_start_token'],
                       x['label_end_token'])
                       ),
            input_generator)
        label_count = 2
    elif task == 'short_ans_yesno':
        formatted_result_generator = map(
            lambda x: (input_data_converter(x),
                       x['label_yes_no']
                       ),
            input_generator)
        label_count = 1
    else:
        pass
    # format fetched in the first run
    if label_count == 1:
        dataset = tf.data.Dataset.from_generator(
            lambda: formatted_result_generator,
            tuple([input_tf_types, tf.int32]),
            tuple([input_tf_shapes, tf.TensorShape([])]),
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            lambda: formatted_result_generator,
            tuple([input_tf_types, tuple([tf.int32]*label_count)]),
            tuple([input_tf_shapes, tuple([tf.TensorShape([])]*label_count)]),
        )
    if 'short_' in task:
        dataset = dataset.cache() 
    dataset = (dataset
        .shuffle(buffer_size=shuffling_buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
        )
    return dataset
