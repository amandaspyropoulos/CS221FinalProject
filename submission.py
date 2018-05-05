#!/usr/bin/python
# Amanda Spyropoulos


import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    dict = collections.defaultdict(int)
    for word in x.split():
        dict[word] += 1
    return dict
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    # go through all points in the train data
    def predictor(x):
        if(dotProduct(weights, featureExtractor(x)) > 0):
            return 1;
        return -1;

    for i in range(0, numIters):
        for data in trainExamples:
            x = data[0]
            y = data[1]
            phi = featureExtractor(x)
            if((1 - dotProduct(weights, phi)*y) > 0):
                increment(weights, eta*y, phi);
        print evaluatePredictor(trainExamples, predictor) 
    
    # END_YOUR_CODE
    return weights

    


############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = collections.defaultdict(int)

        for key in weights.keys():
            if(random.randint(0,1) %2 == 0):
                phi[key] = random.randint(-100, 100) # for now, just pick some arbitrary max value
        
        if(dotProduct(weights, phi) > 0):
            return (phi, 1)
        else:
            return (phi, -1)
        # END_YOUR_CODE
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):

        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x_no_spaces = "".join(x.split(" "))
        nGrams = collections.defaultdict(int)
        for i in range(0, len(x_no_spaces) - n + 1):
            nGrams[x_no_spaces[i:i+n]] += 1
        return nGrams;

        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):

    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    threshold = 0.001
    centroids = [] # a list of dicts
    clusters = [] # a list of lists of dicts

    # initialize cluster centroids to be random elements of examples
    for i in range (0, K):
        example = random.choice(examples) 
        centroids.append(example)
        clusters.append([])

    totalCost = 0
    prevCost = 0

    # precalculate example*example

    exampleCache = collections.defaultdict(float)

    for index, example in enumerate(examples):
        exampleCache[index] = dotProduct(example, example)

    # for up to maxIters iterations
    for currentIter in range(0, maxIters):
        totalCost = 0 # reset to totalCost

        # precalculate centroid*centroid
        centroidCache = collections.defaultdict(float)

        for centroidIndex, c in enumerate(centroids):
            centroidCache[centroidIndex] = dotProduct(c, c)

        # go through each example and assign each to a centroid
        for exampleIndex, example in enumerate(examples):
            minLoss = float("inf")
            minCentroidIndex = -1

            for index, centroid in enumerate(centroids):
                # loss function
                loss = exampleCache[exampleIndex] + centroidCache[index] - 2*dotProduct(example, centroid)
                
                if minLoss >= loss: 
                    minLoss = min(minLoss, loss)
                    minCentroidIndex = index
            clusters[minCentroidIndex].append(example) # add the example to the cluster with the lowest loss
            totalCost += minLoss
        if(abs(prevCost - totalCost) < threshold):
            break
        # then, recalculate all of the centroids and clear the clusters if we're repeating k-means again
        newCentroids = []
        for index, centroid in enumerate(centroids):
                newCentroid = {}

                for clusterPoint in clusters[index]:
                    increment(newCentroid, 1.0/(len(clusters[index])), clusterPoint)
                
                newCentroids.append(newCentroid) 
        
        prevCost = totalCost
        # ok clear clusters
        clusters = [[] for i in range(K)]
        centroids = newCentroids

    # now we've converged or finished iterating, so return!

    return centroids, clusters, totalCost

    # END_YOUR_CODE





