from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix


def multi_class():
    mnist = load_digits()
    # print(mnist)
    target = mnist["target"]
    images = mnist["images"]
    # flatten the images
    n_samples = len(images)
    flatten_images = images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    # clf = svm.SVC(gamma=0.001)
    sgd_clf = SGDClassifier(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        flatten_images, target, test_size=0.3, shuffle=False
    )

    # Learn the digits on the train subset
    # clf.fit(X_train, y_train)
    sgd_clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    # predicted = clf.predict(X_test)
    predicted = sgd_clf.predict(X_test)
    print(predicted)
    print(y_test)
    print("confusion_matrix: ", confusion_matrix(y_test, predicted))
    print("Precision Score : ", precision_score(y_test, predicted, average='micro'))
    print("Recall Score : ", recall_score(y_test, predicted, average='micro'))


def binnaryClass():
    mnist = load_digits()
    # print(mnist)
    target = mnist["target"]
    images = mnist["images"]
    # flatten the images
    n_samples = len(images)
    flatten_images = images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    # clf = svm.SVC(gamma=0.001)
    sgd_clf = SGDClassifier(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        flatten_images, target, test_size=0.3, shuffle=True
    )
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    # Learn the digits on the train subset
    # clf.fit(X_train, y_train)
    sgd_clf.fit(X_train, y_train_5)

    # Predict the value of the digit on the test subset
    # predicted = clf.predict(X_test)
    predicted = sgd_clf.predict(X_test)
    print(predicted)
    print(y_test_5)
    print("confusion_matrix: ", confusion_matrix(y_test_5, predicted))
    print("Precision Score : ", precision_score(y_test_5, predicted, average='micro'))
    print("Recall Score : ", recall_score(y_test_5, predicted, average='micro'))


binnaryClass()
# def show_img(img):
#     plt.gray()
#     plt.matshow(img)
#
#     plt.show()
#
#
# def reshape3d_to_2d(matrix):
#     samples, nx, ny = matrix.shape
#     # return matrix.reshape((samples, nx * ny))
#     return matrix.reshape(-1, nx*ny*1)
#
#
# mnist = load_digits()
# # print(mnist)
# data = mnist["data"]
# target = mnist["target"]
# images = mnist["images"]
# train_partition = int(len(images) * 0.7)
# flat_images = reshape3d_to_2d(images)
# X_train, X_test, y_train, y_test = \
#     flat_images[:train_partition], flat_images[train_partition:], target[:train_partition], target[train_partition:]
#
# shuffle_index = np.random.permutation(train_partition)
# X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
# y_train_5 = (y_train == 5)
# y_test_5 = (y_test == 5)
#
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)
# #
# val = sgd_clf.predict(X_test[0])
# print(val)
#
# show_img(X_test[0])
#
#
# # ind = 38
# # print(images[ind].shape)
#
#
#

'''
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier


def show_img(img):
    plt.gray()
    plt.matshow(img)

    plt.show()


def reshape3d_to_2d(matrix):
    samples, nx, ny = matrix.shape
    # return matrix.reshape((samples, nx * ny))
    return matrix.reshape(-1, nx*ny*1)


mnist = load_digits()
# print(mnist)
data = mnist["data"]
target = mnist["target"]
images = mnist["images"]
train_partition = int(len(images) * 0.7)
X_train, X_test, y_train, y_test = \
    images[:train_partition], images[train_partition:], target[:train_partition], target[train_partition:]

shuffle_index = np.random.permutation(train_partition)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
d2_train = reshape3d_to_2d(X_train)
print(d2_train)
d2_test = reshape3d_to_2d(X_test)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(d2_train, y_train_5)
#
val = sgd_clf.predict(X_test[0])
print(val)

show_img(X_test[0])


# ind = 38
# print(images[ind].shape)

'''
