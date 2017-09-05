# coding:utf-8
from numpy import *
def loadData():
    users_items = {'A':{'a':1,'b':1,'d':1},
                'B':{'a':1,'c':1},
                'C':{'b':1,'e':1},
                'D':{'c':1,'d':1,'e':1}}
    return users_items

# load movie data
def loadDataSet(fileName):
    fr = open(fileName)
    # create dictionary
    user_items = dict()
    for line in fr.readlines():
        user,item,score,time = line.strip().split("\t")
        # user-item {user : {}}
        user_items.setdefault(int(user), {})
        # {user:{item:score}}
        user_items[int(user)][int(item)] = int(score)
    return user_items

# calculate user's similary
def userSimilarity(user_items):
    # {item:users{}}
    item_users = dict()
    for user,items in user_items.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(user)
    C = dict()
    N = dict()
    for i,users in item_users.items():
        for u in users:
            N.setdefault(u, 0)
            N[u] += 1
            C.setdefault(u,{})
            for v in users:
                if u == v:
                    continue
                C[u].setdefault(v,0)
                C[u][v] += 1
    W = dict()
    for u,related_users in C.items():
        W.setdefault(u,{})
        for v,cuv in related_users.items():
            W[u][v] = cuv/math.sqrt(N[u]*N[v])
    return W,C,N

def recommend(user_items, userSimilarity, user, k=3, N=10):
    rank = dict()
    # print("user_items",user_items)
    # print("size : " , user_items)
    # print("user_items[user]", user_items[user])
    action_item = user_items[user].keys()
    for v,wuv in sorted(userSimilarity[user].items(), key = lambda x:x[1], reverse = True)[0:k]:
        for i,rvi in user_items[v].items():
            if i in action_item:
                continue
            rank.setdefault(i,0)
            rank[i] += wuv*rvi
    return dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[0:N])


if __name__ == '__main__':
    user_items = loadDataSet("/Users/scofield/MLRep/Data/CollaborativeFilter.data")
    # print("user_items",user_items)
    userSimilarit,C,N = userSimilarity(user_items)
    userTop = 3
    recommendResult = recommend(user_items, userSimilarit, userTop)
    print("recommendResult : ", recommendResult)
