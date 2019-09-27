#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:47:35 2019

@author: saurabh
"""
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """For 2019, Glassdoor ranked data scientist as the number one best job in America. The 
    ever-growing popularity of the data science career path has led to an explosion in degrees, programs, 
    and bootcamps targeted at people looking to enter the field. According to Discover Data Science 
    a bachelors degree in data science was nearly non-existent 5 years ago and now there are over 50. 
    While demand for data scientists continues to grow (Indeed reported a 29% increase in demand from 
    2018 to 2019), I would argue that landing your first job as a data scientist is perhaps harder than 
    ever due to the increased supply of entry-level talent. It doesn’t help that many job descriptions 
    express qualifications around having multiple years of experience in the field. In fact, the first 
    description I found for a data scientist job on Indeed had the following (note: this might be 
    completely appropriate for this particular job; using it to illustrate the point that there tend to be
    few data science job descriptions that don’t require multiple years of experience):

    Bachelor’s degree or four or more years of work experience.
    Four or more years of relevant work experience.
    Given all of this, it is not surprising that one of the most common questions I receive is, “How do I break into the field of data science?”

    There are thousands of ways one might answer this question, but I’d like to focus on four. My 
    answers assume a bit about you, though. Primarily, that you have in fact studied data science and 
    feel comfortable with the basic principles of the field. If you are not sure, check out Andrew Ng’s 
    machine learning course. If you feel comfortable with that material then I believe my advice applies 
    (bonus points if you have also taken his deep learning course)."""

lemmatize = WordNetLemmatizer()   
sentence = nltk.sent_tokenize(paragraph)

for i in range(len(sentence)):
    sentence[i] = re.sub('[^a-zA-Z]',' ',sentence[i])
    word = nltk.word_tokenize(sentence[i])
    word = [lemmatize.lemmatize(j).lower() for j in word if j.lower() not in set(stopwords.words('english'))]
    sentence[i] = ' '.join(word)

#creating the bag of word model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(sentence).toarray()

















