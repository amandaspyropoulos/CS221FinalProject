# works just hella inefficient

'''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    threshold = 0.01

    centroids = [] # a list of dicts
    clusters = [] # a list of lists of dicts

    # initialize cluster centroids to be random elements of examples
    for i in range (0, K):
        example = random.choice(examples) 
        centroids.append(example)
        clusters.append([])

    totalCost = 0
    prevCost = 0

    # for up to maxIters iterations
    for currentIter in range(0, maxIters):
        totalCost = 0 # reset to totalCost

        # go through each example and assign each to a centroid
        for example in examples:
            minLoss = float("inf")
            minCentroidIndex = -1

            for index, centroid in enumerate(centroids):
                # loss function
                ex = dict(example)
                increment(ex, -1, centroid)
                loss = dotProduct(ex, ex)

                
                if minLoss >= loss: 
                    minLoss = min(minLoss, loss)
                    minCentroidIndex = index
            clusters[minCentroidIndex].append(example) # add the example to the cluster with the lowest loss
            totalCost += minLoss
        if(prevCost == totalCost):
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