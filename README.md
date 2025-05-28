# Installation

Our program is implemented based on the open-source framework RecBole.

RecBole requires Python 3.7 or later to run.

RecBole requires torch version 1.7.0 or later. If you want to run RecBole on a GPU, ensure that your CUDA version or CUDAToolkit version is 9.2 or later. This requires your NVIDIA driver version to be 396.26 or later (on Linux) or 397.44 or later (on Windows 10).

**Install from Conda**

```
conda install -c aibox recbole

```

**Install from pip**

```
pip install recbole

```

**Install from source**

```
git clone '<https://github.com/RUCAIBox/RecBole.git>' && cd RecBole
pip install -e . --verbose

```

## Data Processing:

This paper uses four datasets:

1. **MovieLens** ('https://grouplens.org/datasets/movielens/')
    
    It contains user ratings on movies. We use the 1M version and treat all rated movies as interactive items.
    
2. **Yelp** ('https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/download?datasetVersionNumber=1')
    
    It contains user reviews of restaurants and bars. We use the transaction records after Jan. 1st, 2018.
    
3. **Gowalla** ('https://snap.stanford.edu/data/loc-gowalla.html')
    
    This is a check-in dataset obtained from Gowalla, where users share their locations by checking in.
    
4. **Foursquare** ('https://www.kaggle.com/chetanism/foursquare-nyc-and-tokyo-checkin-dataset')
    
    This dataset contains check-ins in NYC and Tokyo collected over about 10 months.
    

To use RecBole, you need to convert these raw datasets into atomic files, a data format defined by RecBole. Download the raw datasets and process them using the conversion tools provided in this repository. You can find detailed steps in `conversion_tools/usage/`.

## Running

If you want to run a model, simply execute:

```
python run_recbole.py --model=PNN

```

In this mode, it will run as described in Section 4.4 of the article.

If you want to change parameters, such as `learning_rate` or `embedding_size`, just add additional parameters according to your needs, for example:

```
python run_recbole.py --learning_rate=0.0001 --embedding_size=128

```

If you want to change the model being run, just add additional settings when executing the script:

```
python run_recbole.py --model=[model_name]
```
