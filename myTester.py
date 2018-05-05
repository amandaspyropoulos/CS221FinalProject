# Amanda Spyropoulos
# v1, 5/4/18
import re
import time
from util import *
import submission
import random
import collections
import math
import sys
from util import *
import pandas as pd
import numpy as np


if __name__ == '__main__': 
	#### UNCOMMENT THIS SECTION after you run this for the first time

	#Create the test and train .dev files
	# create a new file
	totalRows = 159572
	testingRatio = .1

	fTrain = open("CommentTrain.dev","a+")
	fTest = open("CommentTest.dev","a+")

	# let's read the csv file

	data = pd.read_csv("train.csv")

	for index, row in data.iterrows():
		if index > totalRows*(1 - testingRatio): # set aside some of the data for testing
			f = fTest
		else:
			f = fTrain

		comment = row['comment_text']
		rating = row['toxic'] or row['severe_toxic'] or row['obscene'] or row['threat'] or row['insult'] or row['identity_hate']
		if not rating:
			f.write("+1 %s\r\n" % comment)
		else:
			f.write("-1 %s\r\n" % comment)
	f.close()
	
	#####

	negativeThreshold = -.5

	trainExamples = readExamples('CommentTrain.dev')
	devExamples = readExamples('CommentTest.dev')
	featureExtractor = submission.extractWordFeatures
	weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
	outputWeights(weights, 'weights')
	outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
	trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
	devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
	print "Official: train error = %s, dev error = %s" % (trainError, devError)
    
	toxicExamples = readExamples('toxic.dev')
	for example in toxicExamples:
		comment, rating = example
		words = comment.split(' ')
		result = ""
		for word in words:
			# remove any non-alphabetic characters
			newWord = re.sub("[^a-zA-Z]","", word)
			newWord = newWord.lower()
			if newWord not in weights or weights[newWord] >= negativeThreshold:
				result += word + " "
		print "Before: %s\n After Detox: %s\n" %(comment, result)

