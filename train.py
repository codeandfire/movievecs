"""A script to train vectors and biases for users and movies."""

import argparse
from collections import namedtuple
from bisect import bisect_left
import csv
import logging
import os
import pickle
import random

import numpy as np
from tqdm import tqdm


# arguments
# ---------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-d', '--dim', type=int, default=20, help='dimensionality of learnt vectors')
parser.add_argument(
    '-e', '--epochs', type=float, default=1.0, help='training epochs (a fraction is also allowed)')
parser.add_argument(
    '-s', '--test-split', type=float, default=0.25, help='fraction of instances in test split')
parser.add_argument(
    '--track-loss', type=int, default=None, help='track stochastic loss after every given number of iterations')
parser.add_argument(
    '--eta-bu', type=float, default=0.1, help='learning rate for user biases')
parser.add_argument(
    '--eta-bm', type=float, default=0.1, help='learning rate for movie biases')
parser.add_argument(
    '--eta-vu', type=float, default=0.1, help='learning rate for user vectors')
parser.add_argument(
    '--eta-vm', type=float, default=0.1, help='learning rate for movie vectors')
parser.add_argument(
    '--lambda-bu', type=float, default=0.01, help='regularization penalty for user biases')
parser.add_argument(
    '--lambda-bm', type=float, default=0.01, help='regularization penalty for movie biases')
parser.add_argument(
    '--lambda-vu', type=float, default=0.01, help='regularization penalty for user vectors')
parser.add_argument(
    '--lambda-vm', type=float, default=0.01, help='regularization penalty for movie vectors')
args = parser.parse_args()

if args.track_loss is not None:
    logging.basicConfig(filename='train.log', filemode='w', level=logging.INFO)


# rating data
# -----------

Rating = namedtuple('Rating', ['user', 'movie', 'value'])

train_data, test_data = [], []
s = 1

print('Loading rating data ... ')
rating_dir = 'dataset'

files = os.listdir(rating_dir)
files_with_progress_bar = tqdm(files, desc='files')

for filename in files_with_progress_bar:

    movie = filename
    movie = movie[:movie.index('.txt')]   # remove the .txt extension

    with open(os.path.join(rating_dir, filename), 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            user, value = row[0], int(row[1])
            rating = Rating(user, movie, value)
            
            # random seed for reproducibility
            # a different random seed for each sample so that the outcome
            # is different (but still reproducible) each time
            random.seed(20 * s)
            s += 1

            if random.random() <= args.test_split:
                test_data.append(rating)
            else:
                train_data.append(rating)


print('{} total.'.format(len(train_data) + len(test_data)))
print(
    'Split into training data with {} ratings and test data with {} ratings.'.format(len(train_data), len(test_data)))

movie_names = sorted(set(r.movie for r in train_data + test_data))
user_ids = sorted(set(r.user for r in train_data + test_data))
M, U = len(movie_names), len(user_ids)


# training
# --------

mu = np.mean([r.value for r in train_data])
user_biases, movie_biases = np.zeros((U,)), np.zeros((M,))

np.random.seed(10)
user_vecs = np.random.randn(U, args.dim) * 0.01

np.random.seed(20)
movie_vecs = np.random.randn(M, args.dim) * 0.01

epoch = 0
train, calc_train_rmse, calc_test_rmse = True, False, False

s = 1

# adagrad memory matrices
mem_user_biases = np.zeros_like(user_biases)
mem_movie_biases = np.zeros_like(movie_biases)
mem_user_vecs = np.zeros_like(user_vecs)
mem_movie_vecs = np.zeros_like(movie_vecs)

print('Performing stochastic gradient descent (with adagrad update) ...')

while True:

    if train:
        # parameters are being trained
        data = train_data

        if epoch == args.epochs:
            # training is done, move to calculating training RMSE
            calc_train_rmse = True
            train = False
            rmse = 0.0

            if args.track_loss is not None:
                print('Loss logged to train.log.')

            print('Calculating training RMSE ...')

        elif args.epochs - epoch < 1.0:
            # less than one epoch of training left
            iters = int((args.epochs - epoch) * len(data))
            data = data[:iters]
            epoch = args.epochs

        else:
            # one more epoch of training
            epoch += 1

        # random seed for reproducibility
        # a different random seed for each epoch so that the shuffling
        # is different (but still reproducible) each time
        random.seed(10 * s)
        s += 1

        random.shuffle(data)


    elif calc_train_rmse:
        # training RMSE has been calculated
        # rather training SSE has been calculated, convert it to RMSE
        train_rmse = np.sqrt(rmse / len(train_data))
        print(f'Training RMSE: {train_rmse:.6f}')

        # move to calculating test RMSE
        data = test_data
        calc_test_rmse = True
        calc_train_rmse = False
        rmse = 0.0
        print('Calculating test RMSE ...')

    elif calc_test_rmse:
        # test RMSE - rather SSE - has been calculated, convert it to RMSE
        test_rmse = np.sqrt(rmse / len(test_data))
        print(f'Test RMSE: {test_rmse:.6f}')
        break


    data_with_progress_bar = tqdm(
        enumerate(data),
        desc='iterations', total=len(data)
    )

    for idx, rating in data_with_progress_bar:

        # bisect_left performs binary search, which is much, much faster than
        # calling .index() on a list and performing linear search.
        u = bisect_left(user_ids, rating.user)
        m = bisect_left(movie_names, rating.movie)

        true = rating.value    # true rating value

        # predicted rating value
        pred = (mu + user_biases[u] + movie_biases[m]
                + np.dot(user_vecs[u], movie_vecs[m]))

        loss = (true - pred) ** 2

        if calc_train_rmse or calc_test_rmse:
            rmse += loss
            continue

        # calculated gradients
        grad_user_bias = (pred - true) + (args.lambda_bu * user_biases[u])
        grad_movie_bias = (pred - true) + (args.lambda_bm * movie_biases[m])
        grad_user_vec = (pred - true) * movie_vecs[m] + (args.lambda_vu * user_vecs[u])
        grad_movie_vec = (pred - true) * user_vecs[u] + (args.lambda_vm * movie_vecs[m])

        # update memory matrices
        mem_user_biases[u] += grad_user_bias ** 2
        mem_movie_biases[m] += grad_movie_bias ** 2
        mem_user_vecs[u] += grad_user_vec ** 2
        mem_movie_vecs[m] += grad_movie_vec ** 2

        # adagrad updates
        user_biases[u] -= args.eta_bu / np.sqrt(mem_user_biases[u] + 1e-8) * grad_user_bias
        movie_biases[m] -= args.eta_bm / np.sqrt(mem_movie_biases[m] + 1e-8) * grad_movie_bias
        user_vecs[u] -= args.eta_vu / np.sqrt(mem_user_vecs[u] + 1e-8) * grad_user_vec
        movie_vecs[m] -= args.eta_vm / np.sqrt(mem_movie_vecs[m] + 1e-8) * grad_movie_vec

        if args.track_loss is not None:
            if (idx + 1) % args.track_loss == 0:
                logging.info(f'epoch {epoch} iteration {idx + 1}: loss {loss:.6f}')


# save parameters
# ---------------

with open('params.pkl', 'wb') as f:
    pickle.dump(
        {
            'mu': mu,
            'user_biases': user_biases,
            'movie_biases': movie_biases,
            'user_vecs': user_vecs,
            'movie_vecs': movie_vecs,
            'user_ids': user_ids,
            'movie_names': movie_names
        }, f)

print('Trained parameters saved to params.pkl')
