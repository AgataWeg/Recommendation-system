###########################################
### CONTENT BASED - MEMORY BASED SYSTEM ###
###########################################

from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from collections import defaultdict
from operator import itemgetter
import heapq
import pandas as pd
import numpy as np
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity


### possible varaibles to change ###

# DIR1 is a file where is situated data: categories and names of films given in csv format:
# movieId,title,genres

# DIR2 is a file where is situated data: users ratings given in csv format:
# userId,movieId,rating

# test_subject is  the person id for who to recommend films
# k is the parameter how many best ratings take for further predictions
# M - how many films to recommend

# the categories are given in cat variable

DIR1 = "filePathMoviesWithgenres.csv"
DIR2 = "filePathUsersWithRating.csv"
test_subject = "10"
k = 15
M = 10
print("Recommendations for User", test_subject)
### categories ###
# 18 possible categories, if another - to change
cat = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "FilmNoir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "SciFi",
    "Thriller",
    "War",
    "Western",
]


### load names and categories ###
def load_datasetCategoriesAndNames(fileName):
    movieID_to_name = {}
    movie_category = {}
    cat_def_num = defaultdict()
    temp = {}

    with open(fileName, newline="", encoding="ISO-8859-1") as csvfile:
        movie_reader = csv.reader(csvfile)
        next(movie_reader)
        for row in movie_reader:
            movieID = int(row[0])
            movie_name = row[1]
            movie_category[movieID] = row[2].split("|")
            # changing categories names to intiger array
            temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for el in movie_category[movieID]:
                l = 0
                for k in cat:
                    if k == el:
                        temp[l] = 1
                        break
                    else:
                        l += 1
            cat_def_num[movieID] = temp
            movieID_to_name[movieID] = movie_name

    return (
        cat_def_num,
        movieID_to_name,
    )


category, movieID_to_name = load_datasetCategoriesAndNames(DIR1)


### load user ratings ###
def load_datasetUserRatings(fileName):
    reader = Reader(line_format="user item rating", sep=",", skip_lines=1)
    ratings_dataset = Dataset.load_from_file(fileName, reader=reader)

    return ratings_dataset


dataset = load_datasetUserRatings(DIR2)

### get top k rated items ###
# building training set from dataset - then taking top k items user rated
trainset = dataset.build_full_trainset()
test_subject_iid = trainset.to_inner_uid(test_subject)
test_subject_ratings = trainset.ur[test_subject_iid]
k_neighbors = heapq.nlargest(k, test_subject_ratings, key=lambda t: t[1])


### returns MovieName ###
def getMovieName(movieID):
    if int(movieID) in movieID_to_name:
        return movieID_to_name[int(movieID)]
    else:
        return ""


### computation for cosine similarities between the best rated and the rest of items ###
candidates = defaultdict(float)
print("Watched before (", k, "the best rated):")

for itemIDn, rating in k_neighbors:
    itemID = trainset.to_raw_iid(itemIDn)
    print(
        "Movie: ",
        getMovieName(int(itemID)),
        "Gatunek:",
        category[int(itemID)],
    )
    try:
        # cosine similarity
        sim_array = {}
        for filmId in list(category.keys()):
            sim_array[filmId] = cosine_similarity(
                [category[filmId]],
                [category[int(itemID)]],
                dense_output=True,
            )
        # value of prediction for each film - rating_of_bestRated x similarity
        for innerID in list(sim_array.keys()):
            candidates[innerID] += sim_array[innerID] * (rating / 5.0)
    except:
        print("can't read from similarity array")

### dictionary of movies that user has watched before ###

watched = {}
for itemIDn, rating in trainset.ur[test_subject_iid]:
    itemID = trainset.to_raw_iid(itemIDn)
    watched[itemID] = 1

### sorting candidates for recommendations and choosing the best ones ###
recommendations = []
position = 1
for itemID, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    # checking whether film hasn't been watched before
    if not itemID in watched:
        recommendations.append(getMovieName(itemID))
        position += 1
        if position > M:
            break

### chosen recommendations printed ###
print("Recommended:")
for r in recommendations:
    print("Movie: ", r)
