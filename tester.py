
if __name__ == '__main__': 
	mdp = submission.BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,threshold=40, peekCost=1)
	startState = mdp.startState()
    alg = util.ValueIteration()
    alg.solve(mdp, .0001)