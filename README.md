This code trains a matrix factorization model that learns to predict the rating (as stars from 1-5) given by a user to a particular movie.

### Model

The model implemented is a very basic matrix factorization model that has the following form:

<a href="https://www.codecogs.com/eqnedit.php?latex=f(u,&space;m)&space;=&space;\mu&space;&plus;&space;b_u&space;&plus;&space;b_m&space;&plus;&space;\vec{u}^T&space;\vec{m}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(u,&space;m)&space;=&space;\mu&space;&plus;&space;b_u&space;&plus;&space;b_m&space;&plus;&space;\vec{u}^T&space;\vec{m}" title="f(u, m) = \mu + b_u + b_m + \vec{u}^T \vec{m}" /></a>

Essentially, this model predicts the rating given by a user 'u' to a movie 'm' by learning two vectors, one for 'u' and one for 'm', and taking their dot product (along with adding a couple of bias terms). Hence the name `movievecs`: vectors are learnt for movies (as well as users).

### Requirements

  - Bash shell (on Windows you can use Git Bash),
  - Python (any 3.x version should work), as well as Numpy (and `tqdm`, to display nice progress bars):
    ```
    $ pip install numpy tqdm
    ```

### Usage

  1. Download the [Netflix Prize data](https://www.kaggle.com/netflix-inc/netflix-prize-data). This dataset consists of real ratings given on Netflix, by over half a million users to 17770 movies. The whole dataset is about 2 GB in size. Verify that it contains the following files:
     ```
     combined_data_1.txt
     combined_data_2.txt
     combined_data_3.txt
     combined_data_4.txt
     movie_titles.csv
     probe.txt
     qualifying.txt
     README
     ```
  3. Create a directory called `netflix` and move all these files to that directory. (Actually, only the first five files are required.)
  4. Run the script `extract.sh` to extract the data for movies of your choice from this Netflix dataset. For example
     ```
     $ bash extract.sh 'apollo 13'
     ID,Year,Title
     7745,1995,Apollo 13
     Confirm ID 7745? [y/n] y
     Ratings saved to apollo_13_7745.txt 
     ```
  5. Create a directory called `dataset` and move all the `.txt` files containing the extracted data to that directory.
  6. Finally, run `train.py` to train the model. A sample output of this script is shown below:
     ```
     $ python3 train.py
     Loading rating data ... 
     files: 100%|███████████████████████████████████| 36/36 [00:37<00:00,  1.04s/it]
     3742159 total.
     Split into training data with 2805813 ratings and test data with 936346 ratings.
     Performing stochastic gradient descent (with adagrad update) ...
     iterations: 100%|█████████████████| 2805813/2805813 [01:46<00:00, 26392.53it/s]
     Calculating training RMSE ...
     iterations: 100%|████████████████| 2805813/2805813 [00:16<00:00, 168617.65it/s]
     Training RMSE: 0.793794
     Calculating test RMSE ...
     iterations: 100%|██████████████████| 936346/936346 [00:05<00:00, 174716.14it/s]
     Test RMSE: 0.919558
     Trained parameters saved to params.pkl
     ```
  7. You can load the trained parameters using Python's `pickle` module:
     ```python
     import pickle
     with open('params.pkl', 'rb') as f:
         params = pickle.load(f)
     ```
`params` is a Python dictionary, containing the parameters of the model (`mu`, `user_biases`, `movie_biases`, `user_vecs`, `movie_vecs`) as well as a list of all the movie names (`movie_names`) and user IDs (`user_ids`).
```
>>> params.keys()
dict_keys(['mu', 'user_biases', 'movie_biases', 'user_vecs', 'movie_vecs', 'user_ids', 'movie_names'])
```
To predict the `rating` for a user given by ID `user` and a movie given by its name `movie`, you can do
```python
import numpy as np
u, m = params['user_ids'].index(user), params['movie_names'].index(movie)
rating = (
    params['mu'] + params['user_biases'][u] + params['movie_biases'][m] + 
    np.dot(params['user_vecs'][u], params['movie_vecs'][m])
)
```
[Here](clustering/clustering.md), I discuss a clustering experiment on the movie vectors which yields very interesting results.

More details on both the scripts (`extract.sh` and  `train.py`) are given in the [Details](#details) section below.


### Time

The `train.py` script is quite fast. On a PC with 8 GB RAM and a 4-core CPU, given the extracted data of 36 movies that contains about 3.7 million ratings in total, this script takes about 3 minutes to run.


### Details

`extract.sh` takes a search phrase as argument and looks through the file `movie_titles.csv` for matches. Ensure you use proper quoting when the phrase involves multiple words, for example `'apollo 13'`. When there is only one match, it simply prompts the user for a confirmation of the movie's ID:
```
$ bash extract.sh 'apollo 13'
ID,Year,Title
7745,1995,Apollo 13
Confirm ID 7745? [y/n] y
Ratings saved to apollo_13_7745.txt 
```
but when there are more matches, it prompts the user to manually enter the correct ID:
```
$ bash extract.sh 'armageddon'
ID,Year,Title
69,2003,WWE: Armageddon 2003
621,1997,Armageddon
6972,1998,Armageddon
8180,1993,Warlock: The Armageddon
9322,1979,Doctor Who: The Armageddon Factor
13429,1998,Getter Robo Armageddon: Vol. 1: Resurrection
Manually enter ID: 6972
Ratings saved to armageddon_6972.txt
```
The name of the file to which the ratings are saved is derived from the search phrase and the ID selected.

If no matches are found for the search phrase:
```
$ bash extract.sh 'blah blah'
blah blah not found.
```

`train.py` trains the matrix factorization model using stochastic gradient descent with AdaGrad updates. As you can see from a sample output
```
$ python3 train.py
Loading rating data ... 
files: 100%|███████████████████████████████████| 36/36 [00:37<00:00,  1.04s/it]
3742159 total.
Split into training data with 2805813 ratings and test data with 936346 ratings.
Performing stochastic gradient descent (with adagrad update) ...
iterations: 100%|█████████████████| 2805813/2805813 [01:46<00:00, 26392.53it/s]
Calculating training RMSE ...
iterations: 100%|████████████████| 2805813/2805813 [00:16<00:00, 168617.65it/s]
Training RMSE: 0.793794
Calculating test RMSE ...
iterations: 100%|██████████████████| 936346/936346 [00:05<00:00, 174716.14it/s]
Test RMSE: 0.919558
Trained parameters saved to params.pkl
```
this script performs the following:
  1. it loads the extracted rating data (in my case I had extracted the data of 36 movies, which amounted to about 3.7 million ratings),
  2. splits the data into training and test sets (by default it is a 75-25 split),
  3. carries out stochastic gradient descent (one epoch by default),
  4. calculates the RMSE (root-mean-squared error) of the trained model over the training and test sets,
  5. and saves the trained parameters to disk.

The user and movie vectors learnt have a dimensionality of 20 by default. The learning rate of all the biases and vectors is 0.1 by default and the L2-regularization penalty is 0.01.

You can change all of these default values, of course:
```
$ python3 train.py --help
usage: train.py [-h] [-d DIM] [-e EPOCHS] [-s TEST_SPLIT]
                [--track-loss TRACK_LOSS] [--eta-bu ETA_BU] [--eta-bm ETA_BM]
                [--eta-vu ETA_VU] [--eta-vm ETA_VM] [--lambda-bu LAMBDA_BU]
                [--lambda-bm LAMBDA_BM] [--lambda-vu LAMBDA_VU]
                [--lambda-vm LAMBDA_VM]

A script to train vectors and biases for users and movies.

optional arguments:
  -h, --help            show this help message and exit
  -d DIM, --dim DIM     dimensionality of learnt vectors
  -e EPOCHS, --epochs EPOCHS
                        training epochs (a fraction is also allowed)
  -s TEST_SPLIT, --test-split TEST_SPLIT
                        fraction of instances in test split
  --track-loss TRACK_LOSS
                        track stochastic loss after every given number of
                        iterations
  --eta-bu ETA_BU       learning rate for user biases
  --eta-bm ETA_BM       learning rate for movie biases
  --eta-vu ETA_VU       learning rate for user vectors
  --eta-vm ETA_VM       learning rate for movie vectors
  --lambda-bu LAMBDA_BU
                        regularization penalty for user biases
  --lambda-bm LAMBDA_BM
                        regularization penalty for movie biases
  --lambda-vu LAMBDA_VU
                        regularization penalty for user vectors
  --lambda-vm LAMBDA_VM
                        regularization penalty for movie vectors
```
A few of these options probably require more explanation and are described below:
  - The number of epochs can be a fraction, i.e. 0.5 will train for half an epoch and 1.5 will train for one-and-a-half epochs.
  - To change the train-test split, you have to provide a fraction for the test split. For example a test split of 0.3 will result in a 70-30 train-test split.
  - You can track the stochastic loss after every few updates. For example, specifying the number 10000 will report the stochastic loss at the 10000th update, 20000th update, 30000th update and so on. This report is written to a log file `train.log`:
    ```
    $ head train.log 
    INFO:root:epoch 1 iteration 10000: loss 0.429710
    INFO:root:epoch 1 iteration 20000: loss 0.166349
    INFO:root:epoch 1 iteration 30000: loss 0.590381
    INFO:root:epoch 1 iteration 40000: loss 0.587494
    INFO:root:epoch 1 iteration 50000: loss 0.322644
    INFO:root:epoch 1 iteration 60000: loss 1.863725
    INFO:root:epoch 1 iteration 70000: loss 0.588115
    INFO:root:epoch 1 iteration 80000: loss 0.681265
    INFO:root:epoch 1 iteration 90000: loss 1.252593
    INFO:root:epoch 1 iteration 100000: loss 0.169029
    ```
The rest of the options are self-explanatory.
