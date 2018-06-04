# Amanda Spyropoulos
# v3, 6/3/18
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
import nltk
from nltk.corpus import wordnet
from nltk.translate.bleu_score import sentence_bleu

class TranslateProblem(SearchProblem):
	def __init__(self, comment, weights):
		# Comment is a full comment, turned into a list of words
		self.comment = comment
		# Weights is a dictionary where keys are words and vals are their weight, as predicted by our linear classifier
		self.weights = weights

	def startState(self):
		# Return information for start state
		return 0

	def isEnd(self, state):
		return (len(self.comment)) == state

	def succAndCost(self, state):
		index = state
		successors = []

		# Create list of Actions
		#actions = synonyms
		def findSynonyms(word):
			synonyms = []
			for syn in wordnet.synsets(word):
				for l in syn.lemmas():
					synonyms.append(l.name())
			return synonyms
		actions = findSynonyms(self.comment[index])
		actions.append("ACTION_KEEP")
		actions.append("ACTION_DELETE")

		#print(self.comment[index], actions)

		# For each action, find successor state and cost
		for a in actions:
			newState = index + 1
			cost = 0
			if a == "ACTION_KEEP":
				if self.comment[index] in self.weights:
					cost = self.weights[self.comment[index]]
			elif a != "ACTION_DELETE":
				#print(a)
				if a in self.weights:
					cost = self.weights[a]
				else:
					continue
			successors.append((a, newState, cost))
		return successors

def convertUCSActionsToString(words, actions):
	newActions = []

	for index, word in enumerate(actions):
		if word ==  "ACTION_KEEP":
			newActions.append(words[index])
		elif word == "ACTION_DELETE":
			pass
		else:
			newActions.append(str(word))

	return ' '.join(newActions)


if __name__ == '__main__':
	testing = True
	#### UNCOMMENT THIS SECTION after you run this for the first time

	#Create the test and train .dev files
	# create a new file
	totalRows = 159572
	testingRatio = .1

	if testing:
		totalRows = 5000
		print "TESTING"

	fTrain = open("CommentTrain.dev","wa+")
	fTest = open("CommentTest.dev","wa+")

	# let's read the csv file

	data = pd.read_csv("train.csv")

	for index, row in data.iterrows():
		if index > totalRows:
			break
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

	print "done with creating CommentTrain and CommentDev files"
	#####

	negativeThreshold = -.5

	trainExamples = readExamples('CommentTrain.dev')
	devExamples = readExamples('CommentTest.dev')
	featureExtractor = submission.extractWordFeatures
	if testing:
		numIters = 4
	else:
		numIters = 20
	weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, numIters, eta=0.01)
	outputWeights(weights, 'weights')
	outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
	trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
	devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
	print "Official: train error = %s, dev error = %s" % (trainError, devError)

	toxicExamples = readExamples('toxic.dev')
	translatedToxicExamples = readExamples('toxicTranslated.dev')

	print("About to initialize UCS")
	ucs = UniformCostSearch(verbose=0)
	print("Successfully initialized UCS.")
	for index, example in enumerate(toxicExamples):
		comment, rating = example
		
		translatedComment, TranslatedRating = translatedToxicExamples[index]
		
		#comment = example
		comment = re.sub("[^a-zA-Z ]","", comment)
		comment = comment.lower()
		words = comment.split(' ')
		#for word in words:
			# remove any non-alphabetic characters
			#word = re.sub("[^a-zA-Z]","", word)
			#word = word.lower()
		print("About to solve Translate Problem.")
		ucs.solve(TranslateProblem(words, weights))
		print("Successfully solved Translate Problem.\n\n")

		print "Before:\n %s\n After Detox: " % comment

		machineTranslatedComment = convertUCSActionsToString(words, ucs.actions)
		print(machineTranslatedComment)
		score = sentence_bleu(machineTranslatedComment.split(), translatedComment.split())
		print "Human - translated comment: "
		print translatedComment
		print "Score: "
		print (score)


