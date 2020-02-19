# NQA
This repository is for the Kaggle competition of Google's Natural Question Answer

# Prerequest Steps for this project 
1. Install transformers: 
```
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
```
!git clone https://github.com/KWGroup/NQA.git
```
# Tutorial to the Preprocessor package 
[A runnable colab version to this tutorial](https://colab.research.google.com/gist/jeffrey82221/c27c3294fda0ede8092c42785ec86df8/tutorial-to-preprocessor.ipynb) 

## Import the preprocessing functions 
```python
# Import the generator of raw training data 
from NQA.Preprocessor import get_train_data 
# Import the data warper 
from NQA.Preprocessor import create_answer_dataset, create_answer_data_generator
# Import the data formatter 
from NQA.Preprocessor import create_input_output_featureset, input_output_feature_generator
```

## Import the tokenizer 
```python
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
```
## preprocessing for the training of short answer models
### With intermediate dataframe stored 
```python
task = 'short_answer' 
# This variable can also be set as "classing" or "squading" as wish. 
train_data_generator = get_train_data()
# Limit the training data into a subset to avoid long processing time (remove in real case)
training_data_subset = [next(train_data_generator) for _ in range(100)] 
# Saving warped result to a temporary dataframe 
tmp_dataframe = create_answer_dataset(
    training_data_subset,task = task)
# Obtaining the data formatted result dataframe 
preprocessed_dataframe = create_input_output_featureset(tmp_dataframe, tokenizer, task = task)
```

### With intermediate generator used to avoid intermediate storage  
```python
task = 'short_answer' 
# This variable can also be set as "classing" or "squading" as wish. 
train_data_generator = get_train_data()
# Limit the training data into a subset to avoid long processing time (remove in real case)
training_data_subset = [next(train_data_generator) for _ in range(100)] 
# * Create a generator that produces warped training examples 
intermediate_generator = create_answer_data_generator(
    training_data_subset,task = task)
# Obtaining the data formatted result dataframe 
# * by connecting the intermediate_generator to the data formatter 
preprocessed_dataframe = create_input_output_featureset(intermediate_generator, tokenizer, task = task)
```

### Pure generator mode: 

```python
task = 'short_answer' 
# This variable can also be set as "classing" or "squading" as wish. 
train_data_generator = get_train_data()
# Limit the training data into a subset to avoid long processing time (remove in real case)
training_data_subset = [next(train_data_generator) for _ in range(100)] 
# * Create a generator that produces warped training examples 
intermediate_generator = create_answer_data_generator(
    training_data_subset,task = task)
# Obtaining the data formatted result dataframe 
# * by connecting the intermediate_generator to the data formatter 
preprocessed_result_generator = input_output_feature_generator(intermediate_generator, tokenizer, task = task)
```
To create generator that produced preprocced result, simply replace  `create_input_output_featureset` with `input_output_feature_generator`.
## Preprocessing for training the long answer model (i.e., candidate filter)

```python
###############################################################################################################
from NQA.Preprocessor import data_warping_for_candidate_filter
task = 'candidate_filter'
train_data_generator = get_train_data()
training_data_subset = [next(train_data_generator) for _ in range(100)] 
########################################## ^ 79.6 ms per loop #################################################
intermediate_generator = create_answer_data_generator(
    training_data_subset,task = task)
# The additional data warpper: 
intermediate_generator_ = data_warping_for_candidate_filter(intermediate_generator)
########################################## ^ 4.03 s per loop #################################################
preprocessed_result_generator = input_output_feature_generator(intermediate_generator_, tokenizer, task = task)
########################################## ^ 5.17 s per loop #################################################

```
In comparison to the short-answer case, the preprocessing for long answer model (i.e., the candidate filter) requires an additional data warpper to preduct the training instances candidate-by-candidate. 

Here, we recommend using pure generator and avoiding saving any intermediate dataframe, since the amount of training candidate is large.  
