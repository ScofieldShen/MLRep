# coding:utf-8
from numpy import *
def loadData():
    user_items = {
        'A': {'a': 1, 'b': 1, 'd': 1},
        'B': {'b': 1, 'c': 1, 'e': 1},
        'C': {'c': 1, 'd': 1},
        'D': {'b': 1, 'c': 1, 'd': 1},
        'E': {'a': 1, 'd': 1}
    }

    return user_items

def loadDataSet(fileName):
    user_items = dict()
    for line in open(fileName).readlines():
        user,item,score,_ = line.strip().split("\t")
        user_items.setdefault(int(user), {})
        user_items[int(user)][item] = int(score)
    return user_items

def itemSimilarity(user_items):
    C = dict()
    N = dict()
    for user,items in user_items.items():
        for i in items.keys():
            N.setdefault(i, 0)
            N[i] += 1
            C.setdefault(i, {})
            for j in  items.keys():
                if i == j:
                    continue
                C[i].setdefault(j, 0)
                C[i][j] += 1
    W = dict()
    for i,related_items in C.items():
        W.setdefault(i, {})
        for j,cij in related_items.items():
            W[i][j] = cij/(sqrt(N[i]*N[j]))
    return W,C,N

def recommend(user_items, iteSimilarity, user, k=3, N=10):
    rand = dict()
    action_items = user_items[user]
    for item,score in action_items.items():
        for j,wj in sorted(itemSimilarity[item].items(), key=lambda x:x[1], reverse=True)[0:k]:
            if j in action_items.keys():
                continue
            rand.setdefault(j, 0)
            rand[j] += score*wj
    return dict(sorted(rand.items(), key=lambda x:x[1], reverse=True)[0:N])


if __name__ == '__main__':
    user_items = loadDataSet("/Users/scofield/MLRep/Data/CollaborativeFilter.data")
    itemSimilarity,C,N = itemSimilarity(user_items)
    userTopRec = 3
    recommendResult = recommend(user_items, itemSimilarity, userTopRec)
    print("recommendResult : ", recommendResult)