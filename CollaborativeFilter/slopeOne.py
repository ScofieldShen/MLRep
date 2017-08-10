# coding  : utf-8

# slope one
def loadData():
    items = {'A':{1:5,2:3},
           'B':{1:3,2:4,3:2},
           'C':{1:2,3:5}}
    users = {1:{'A':5,'B':3,'C':2},
           2:{'A':3,'B':4},
           3:{'B':2,'C':5}}
    return items,users

# calculate the difference score between items
# users based on user
# items based on item
def buildAveragDiff(users, items, averages):
    # loop each item
    for itemid in items:
        for otherItemId in items:
            # the diff average value of items
            average = 0.0
            # the number of  user both items has been scored
            userRatingPairCount = 0
            # if no difference item
            if itemid != otherItemId:
                # loop each user and item
                for userId in users:
                    # each data is thye score of user to item
                    userRatings = users[userId]
                    if itemid in userRatings and otherItemId in userRatings:
                        userRatingPairCount += 1
                        average += (userRatings[otherItemId] - userRatings[itemid])
                averages[(itemid, otherItemId)] = average/userRatingPairCount
# count the number of people who has the same hobby
def userBothLikeNum(users, itemId1, itemId2):
    count = 0
    for userid in users:
        if itemId1 in users[userid] and itemId2 in users[userid]:
            count += 1
    return count

#***预测评分
#users:用户对物品的评分数据
#items：物品由哪些用户评分的数据
#averages：计算的评分偏差
#targetUserId：被推荐的用户
#targetItemId：被推荐的物品
def suggestedRating(users, items, averages, targetUserId, targetItemId):
    runningRatingCount = 0
    weightedRatingtotal = 0.0
    for i in users[targetUserId]:
        ratingCount = userBothLikeNum(users, i, targetItemId)
        weightedRatingtotal += (users[targetUserId][i] - averages[(targetItemId, i)])\
        * ratingCount
        runningRatingCount += ratingCount
    return weightedRatingtotal/runningRatingCount


if __name__ == '__main__':
    items,users = loadData()
    # print("items : ",items)
    # print("users : ",users)
    averages = {}
    buildAveragDiff(users, items, averages)
    # print("averages : ", averages)

    predictRating = suggestedRating(users, items, averages, 2, 'C')
    print("Guess that user A will rate item3 : " , predictRating)