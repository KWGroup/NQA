# NQA
This repository is for the Kaggle competition of Google's Natural Question Answer

# Prerequest Steps for this project 
1. Install transformers: 
```
!pip install transformers
``` 
2. Download NQA dataset: 
```
import os
os.environ['KAGGLE_USERNAME'] = "XXX" # username of Kaggle platform 
os.environ['KAGGLE_KEY'] = "XXX" # API key in Kaggle's athentication json file 
!kaggle competitions download -c tensorflow2-question-answering
```
3. Switching tensorflow version from 1.X to 2.X: 
```
try:
  %tensorflow_version 2.x
except Exception:
  pass
```
4. Cloning into this repository 
```
!git clone https://github.com/KWGroup/NQA.git
```
