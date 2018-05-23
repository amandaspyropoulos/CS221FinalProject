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

class TranslateProblem(SearchProblem):
	def __init__(self, comment, weights):
		# Comment is a full comment, turned into a list of words
		self.comment = comment.split()
		# Weights is a dictionary where keys are words and vals are their weight, as predicted by our linear classifier
		self.weights = weights

	def startState(self):
		# Return information for start state
		return ([], 0)

	def isEnd(self, state):
		return (len(self.comment)) == state[1]

	def succAndCost(self, state):
		currentCom, index = state
		successors = []

		# Create list of Actions
		#actions = synonyms
		actions = []
		actions.append("ACTION_KEEP")
		actions.append("ACTION_DELETE")

		# For each action, find successor state and cost
		for a in actions:
			newCom = currentCom[:]
			if a == "ACTION_KEEP":
				newCom.append(self.comment[index])
				newState = (newCom, index + 1)
				successors.append((a, newState, self.weights[self.comment[index]]))
			#elif a == "ACTION_DELETE":
				#newState = (newCom, index + 1)
				#successors.append((a, newState, 0)
			else:
				newCom.append(a)
				newState = (newCom, index + 1)
				successors.append((a, newState, self.weights[a]))
		return successors


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
