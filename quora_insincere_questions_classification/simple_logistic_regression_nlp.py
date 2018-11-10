# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import re
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# print(os.listdir(../input))

# regex strings to clean up existing data.
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d)")
REPLACE_WITH_SPACE = re.compile("(\n)|(\-)|(\/)")

# Clean the file using regex defined above and create different lists for ids, messages, classification values. 
# For test set there is empty list of classification.
def clean_split_data(file_name, test_set=False):
    questions_list = []
    targets_list = []
    ids_list = []
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header rows.
        for row in reader:
            # print(row)
            if test_set is False:
                id, question, target_val = tuple(row)
                targets_list.append(target_val)
            else:
                id, question = tuple(row)
            # Clean question with regex.
            question = REPLACE_NO_SPACE.sub("", question.lower())
            question = REPLACE_WITH_SPACE.sub(" ", question.lower())
            questions_list.append(question)
            ids_list.append(id)
    # print("Loading completed.")
    return ids_list, questions_list, targets_list


# Load train data into dataframe.
train_ids_list, train_questions_list, train_targets_list = clean_split_data('../input/train.csv')
test_ids_list, test_questions_list, test_targets_list = clean_split_data('../input/test.csv', test_set=True)
# cv = CountVectorizer(binary=True)
cv = TfidfVectorizer(binary=True)
cv.fit(train_questions_list)
train_transform_data = cv.transform(train_questions_list)
test_transform_data = cv.transform(test_questions_list)
# print("Transform completed.")

# select hyper parameter value of regularization C for logistic regression. If C is less, regularization is high.
train_transform_train_data, train_transform_test_data, target_train, target_test = train_test_split(
    train_transform_data, train_targets_list, train_size=0.75)
# print("Split completed.")
best_f1_score_parameter = 0.01
best_fl_score = 0.0
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(train_transform_train_data, target_train)
    current_fl_score = f1_score(target_test, lr.predict(train_transform_test_data), average='macro')
    if current_fl_score > best_fl_score:
        best_f1_score_parameter = c
    print("F1Score for C=%s: %s" % (c, f1_score(target_test, lr.predict(train_transform_test_data), average='macro')))

# Train the whole model with finalized c value and see the results.
# lr = LogisticRegression(C=0.05)
lr = LogisticRegression(C=best_f1_score_parameter)  # For TF-IDF vectorizer C=1 has better performance, but need to have small value. so satisfied with 0.25
lr.fit(train_transform_data, train_targets_list)
predicted_list = lr.predict(test_transform_data)
print(predicted_list[:10])
#predicted_list = [row.trim() for row in predicted_list]
# print ("Final f1 score C=0.05: %s" % (f1_score(test_targets_list, predicted_list)))
# Write output files.
with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['qid', 'prediction'])
    writer.writerows(zip(test_ids_list, predicted_list))