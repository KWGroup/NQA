# NQA
This repository is for the Kaggle competition of Google's Natural Question Answer

## Outline: 
* [Setups of this project](https://github.com/KWGroup/NQA#setups-of-this-project)
* [Tutorial to the Preprocessor package](https://github.com/KWGroup/NQA#tutorial-to-the-preprocessor-package) 
* [Tutorial to the TFDataset package](https://github.com/KWGroup/NQA#tutorial-to-the-tfdataset-package)
* [Model Training Guide](https://github.com/KWGroup/NQA#Model-Training-Guide)


# Setups of this project 
1. Install transformers: 
```python
!pip install transformers
``` 
2. Download NQA dataset: 
```python
import os
os.environ['KAGGLE_USERNAME'] = "XXX" # username of Kaggle platform 
os.environ['KAGGLE_KEY'] = "XXX" # API key in Kaggle's athentication json file 
!kaggle competitions download -c tensorflow2-question-answering
```
3. Switching tensorflow version from 1.X to 2.X: 
```python
try:
  %tensorflow_version 2.x
except Exception:
  pass
```
4. Cloning into this repository 
```python
!git clone https://github.com/KWGroup/NQA.git
```
# Tutorial to the Preprocessor package 
[A runnable colab version to this tutorial](https://colab.research.google.com/gist/jeffrey82221/c27c3294fda0ede8092c42785ec86df8/tutorial-to-preprocessor.ipynb#scrollTo=W7jC0PxBJGfC) 


## Import the tokenizer 
```python
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
```

## Import the preprocessed data reader 
```python
from NQA.Preprocessor import read_train_dataset
```
This function produces preprocessed result in the form  
of dataframe or generator. 

## Applying the preprocessed data reader  
### (1) Building dataframe from scratch 
```python 
preprocessed_dataframe = read_train_dataset(
      task='candidate_filter',
      mode='build_dataframe',
      tokenizer=tokenizer)
```
### (2) Building data generator 
```python 
preprocessed_instance_generator = read_train_dataset(
      task='candidate_filter',
      mode='build_generator',
      tokenizer=tokenizer)
```
We recommend using `mode == 'build_generator'` for candidate filter task, since 
it takes a large memory space.
### (3) Loading stored dataframe from google drive 
```python 
preprocessed_dataframe = read_train_dataset(
      task='short_ans_entity',
      mode='at_google_drive') 
# when loading from google drive, no need to provide the tokenizer 
```

For the short-answer tasks (i.e., yesno and entity), we recommend using 
`mode='at_google_drive'` and storing a cached preprocessed dataframe in google drive to reduce the time.  

Note that if the code is run on the Kaggle kernal, add argument `on_kaggle=True` to `read_train_dataset`. 
 

## Descomposing the preprocessing into data warping and data formatting 

Our preprocessing method, `read_training_dataset`, is actually built upon 
two steps: 1. data warping, and 2. data formatting. Below is a tutorial 
of how these two steps together processes the raw data for the short and long
answer tasks. 

### Import the data warping and formatting preprocessing functions 
```python
# Import the generator of raw training data 
from NQA.Preprocessor import get_train_data 
# Import the data warper 
from NQA.Preprocessor import create_answer_dataset, create_answer_data_generator
# Import the data formatter 
from NQA.Preprocessor import create_input_output_featureset, input_output_feature_generator
```

### In the case of short answer tasks
####  (1) With intermediate dataframe stored 
```python
task = 'short_answer' 
# This variable can also be set as "short_ans_yesno" or "short_ans_entity" as wish. 
raw_data_generator = get_train_data()
# Saving warped result to a temporary dataframe 
tmp_dataframe = create_answer_dataset(
    raw_data_generator,task = task)
# Obtaining the data formatted result dataframe 
preprocessed_dataframe = create_input_output_featureset(tmp_dataframe, tokenizer, task = task)
```

####  (2) With intermediate generator used to avoid intermediate storage  
```python
task = 'short_answer' 
# This variable can also be set as "short_ans_yesno" or "short_ans_entity" as wish. 
raw_data_generator = get_train_data()
# * Create a generator that produces warped training examples 
intermediate_generator = create_answer_data_generator(
    raw_data_generator,task = task)
# Obtaining the data formatted result dataframe 
# * by connecting the intermediate_generator to the data formatter 
preprocessed_dataframe = create_input_output_featureset(intermediate_generator, tokenizer, task = task)
```

#### (3) Pure generator mode: 

```python
task = 'short_answer' 
# This variable can also be set as "short_ans_yesno" or "short_ans_entity" as wish. 
raw_data_generator = get_train_data()
# * Create a generator that produces warped training examples 
intermediate_generator = create_answer_data_generator(
    raw_data_generator,task = task)
# Obtaining the data formatted result dataframe 
# * by connecting the intermediate_generator to the data formatter 
preprocessed_result_generator = input_output_feature_generator(intermediate_generator, tokenizer, task = task)
```
To create generator that produced preprocced result, simply replace  `create_input_output_featureset` with `input_output_feature_generator`.
### In the case of long answer task (i.e., candidate filtering)

```python
from NQA.Preprocessor import data_warping_for_candidate_filter
task = 'candidate_filter'
raw_data_generator = get_train_data()
intermediate_generator = create_answer_data_generator(
    raw_data_generator,task = task)
# The additional data warpper: 
intermediate_generator_ = data_warping_for_candidate_filter(intermediate_generator)
preprocessed_result_generator = input_output_feature_generator(intermediate_generator_, tokenizer, task = task)

```
In comparison to the short-answer case, the preprocessing for long answer model (i.e., the candidate filter) requires an additional data warpper to produce the training instances candidate-by-candidate. 

Here, we recommend using pure generator and avoiding saving any intermediate dataframe, since the amount of training candidate is large.  

# Tutorial to the TFDataset package 

Functions in this package convert the generator and pd.Dataframe obtained from Preprocessor into a tf.data.Dataset object.

[A runnable colab version to this tutorial](https://colab.research.google.com/drive/1utIbKyBPO3ijnnkqO0mrvm5zhWLNauR-#scrollTo=xDKPv3ioPKxK) 

## (1) Converting a preprocessed dataframe into a tf.data.Dataset 

```python
# Converting the preprocessed dataframe to tf.data.Dataset:
from NQA.TFDataset import df_to_dataset
dataset = df_to_dataset(
    preprocessed_dataframe, batch_size, task = task)
# Testing the resulting dataset
from NQA.TFDataset import dataset_checker
dataset_checker(dataset)
```

## (2) Converting a preprocessed instance generator into a tf.data.Dataset 
```python 
# Converting the preprocessed instance generator to tf.data.Dataset:
from NQA.TFDataset import generator_to_dataset
dataset = generator_to_dataset(preprocessed_result_generator,batch_size, task = task)
# Testing the resulting dataset
from NQA.TFDataset import dataset_checker
dataset_checker(dataset)
```
# Model Training Guide 
Here, we explain how to train our NQA model by using the following parameters.
```python
USE_TPU = False
task = 'short_ans_entity'
batch_size = 256
```
[A runnable colab version to this guide](https://colab.research.google.com/gist/jeffrey82221/90f0a71386d21eb416e60867c07c8f47/model-training-guide.ipynb)

## Setup TPU environment 
```python
import tensorflow as tf
import os
if USE_TPU:
  # create tpu resolver and strategy
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)
```
If you are using the Kaggle kernal, remove `tpu='grpc://' + os.environ['COLAB_TPU_ADDR']` 
from `TPUClusterResolver`. 

## Loading tf.data.Dataset for training 
```python
from NQA.Preprocessor import read_train_dataset 
from NQA.TFDataset import df_to_dataset, generator_to_dataset
import pandas as pd
train_df = read_train_dataset(task=task,mode='at_google_drive')
if type(train_df) == pd.core.frame.DataFrame:
  train_ds = df_to_dataset(train_df,batch_size, 
    task = task
  )
else:
  train_ds = generator_to_dataset(train_df,batch_size, 
    task = task
  )
```
## Training 
### using cpu or gpu 

```python
from NQA.ModelBuilder import create_model 
if not USE_TPU:
  learning_rate = 0.1
  epsilon = 1e-8 
  model = create_model(task)
  optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, 
    epsilon=epsilon
  )
  if task == 'short_ans_entity' or task == 'candidate_filter': 
      model.compile(
        optimizer,
        loss='categorical_crossentropy')
  else:
      model.compile(
        optimizer,
        loss=['categorical_crossentropy','categorical_crossentropy'])
  model.fit(train_ds,verbose = 1)
```

* For CPU, the maximum batch size in 2's exponents is 16. But note that CPU does not speed up computation with larger batch size. Each batch can take > 50 seconds. 
* For GPU, the maximum batch size in 2's exponents is also 16. Each batch takes only about 1s.  

### using TPU


```python
from NQA.ModelBuilder import create_model 
if USE_TPU:
  learning_rate = 0.1
  epsilon = 1e-8 
  with tpu_strategy.scope():
      model = create_model(task)
      optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, 
        epsilon=epsilon
      )
      if task == 'short_ans_entity' or task == 'candidate_filter': 
          model.compile(
            optimizer,
            loss='categorical_crossentropy')
      else:
          model.compile(
            optimizer,
            loss=['categorical_crossentropy','categorical_crossentropy'])
  model.fit(train_ds,verbose = 1)
```
* For TPU, the maximum batch size in 2's exponents is 256. Each batch takes about 0.74 seconds. 
