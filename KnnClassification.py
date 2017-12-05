import numpy as np

def knnClassify(test, train, labels, K):
    threshold=35.0
    test = test+0.0
    train = train+0.0
    [N,M] = train.shape
    #calculate distance between test and training samples
    difference=np.tile(test, (N,1)) - train
    difference=difference**2
    distance=difference.sum(1)
    distance=distance**0.5
    
    #sort the index with decreasing distance
    sortedDisIdx=np.argsort(distance)
    print (np.sort(distance)[1]-np.sort(distance)[0])
    if np.sort(distance)[1]-np.sort(distance)[0]>threshold:
        return "reject"
    #find the k nearest neighbours
    else:
        vote={}
        for i in range(K):
            ith_label=labels[sortedDisIdx[i]]
            vote[ith_label]=vote.get(ith_label,0)+1
            sortedvote=sorted(vote.items(),key=lambda x:x[1], reverse=True)
            return sortedvote[0][0]

def knnClassify_cosSimilarity(test,train,labels,K):
    test = test+0.0
    train = train+0.0    
    [N,M] = train.shape
    #calculate cos similarity between test and training samples
    test = np.tile(test,(N,1))
    dotProduct = (test*train).sum(1)
    test = np.array(test)
    lengthTest = test**2
    lengthTest = lengthTest.sum(1)
    lengthTest = lengthTest**0.5
    lengthTrain = train**2
    lengthTrain = lengthTrain.sum(1)
    lengthTrain = lengthTrain**0.5
    cosSimilarity = dotProduct/(lengthTest*lengthTrain)
    #Sort the index of cosSimilarity
    sortedSimIdx=np.argsort(-cosSimilarity)
    print (np.sort(cosSimilarity)[-1]-np.sort(cosSimilarity)[0])
    #find the k nearest neighbors
    vote={}
    for i in range(K):
        ith_label = labels[sortedSimIdx[i]]
        vote[ith_label] = vote.get(ith_label,0)+1
    sortedvote=sorted(vote.items(),key=lambda x:x[1], reverse=True)
    return sortedvote[0][0]
    