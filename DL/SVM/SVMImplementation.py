from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

def loadData():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people

def getxySampName(lfw_people):
    n_samples, h, w = lfw_people.images.shape
    x = lfw_people.data
    n_features = x.shape[1]
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    return x,y,h,w,n_samples,n_features,n_classes,target_names

#
def getTrainTest(x,y):
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
    return x_train, x_test,y_train,y_test

def getPCA(x_train, x_test, h, w):
    n_components = 150
    print("Extracting the top %d eigenfaces from %d faces" % (n_components, x_train.shape[0]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(x_train)
    print("done in %0.3fs" % (time()-t0))
    eigenfaces = pca.components_.reshape(n_components, h, w)
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    print("done in %0.3fs" % (time() - t0))
    return x_train_pca,x_test_pca,eigenfaces

def svmClassify(x_train_pca,y_train):
    print("Fitting the classifier to the training set : ")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(x_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    return clf

def predictTest(clf,x_test_pca,y_test, target_names,n_classes):
    print("Predicting people's name on test set")
    tt0 = time()
    y_predict = clf.predict(x_test_pca)
    print("done in %0.3s" % (time() - tt0))
    print(classification_report(y_test, y_predict, target_names=target_names))
    print(confusion_matrix(y_test, y_predict, labels=range(n_classes)))
    return y_predict


def title(y_pred,y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted : %s \n true : %s ' % (pred_name, true_name)


def predEvalu(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left = .01, right=.99, top=.90, hspace=.35)
    print("title : ",titles)
    for i in range(n_row * n_col):
        print("i : ", i)
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

if __name__ == '__main__':
    print(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    lfw_people = loadData()
    # print(lfw_people)
    x, y, h, w, n_samples, n_features, n_classes,target_names = getxySampName(lfw_people)
    # print("The total data size : ")
    # print("n_samples %d" % n_samples)
    # print("n_features %d" % n_features)
    # print("n_classes %d" % n_classes)
    x_train, x_test, y_train, y_test = getTrainTest(x, y)
    x_train_pca, x_test_pca,eigenfaces = getPCA(x_train, x_test, h, w)
    clf = svmClassify(x_train_pca, y_train)
    y_pred = predictTest(clf, x_test_pca, y_test, target_names,n_classes)
    pred_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
    predEvalu(x_test, pred_titles, h, w)
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    predEvalu(eigenfaces, eigenface_titles, h, w)
    plt.show()



