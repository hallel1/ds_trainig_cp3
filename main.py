from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    mnist = load_digits()
    target = mnist["target"]
    images = mnist["images"]

    X_train, X_test, y_train, y_test = train_test_split(
        flatten_img(images), target, test_size=0.3, shuffle=True
    )

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    predicted = binary_class(X_train, X_test, y_train_5)
    meause(y_test_5, predicted)

    X_train, X_test, y_train, y_test = train_test_split(
        flatten_img(images), target, test_size=0.3, shuffle=True
    )

    predicted = multi_class_sgd(X_train, X_test, y_train)  # todo: why the results are so bad??
    meause(y_test, predicted)

    # todo: why the result the same of both the training without the split here?
    X_train, X_test, y_train, y_test = train_test_split(
        flatten_img(images), target, test_size=0.3, shuffle=True
    )
    predicted = multi_class_clf(X_train, X_test, y_train)  # todo: why the results are so bad??
    meause(y_test, predicted)


def multi_class_clf(X_train, X_test, y_train):
    print("multi_class_clf")
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)

    return clf.predict(X_test)


def multi_class_sgd(X_train, X_test, y_train):
    print("multi_class_sgd")
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)

    return sgd_clf.predict(X_test)


def binary_class(X_train, X_test, y_train):
    print("binary_class")
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)

    return sgd_clf.predict(X_test)


def meause(y_real, predicted):
    # todo: why prec anf recall are the same always?
    print("Precision Score : ", precision_score(y_real, predicted, average='micro'))
    print("Recall Score : ", recall_score(y_real, predicted, average='micro'))
    no_pred = set(y_real) - set(predicted)
    print(f"no predicts for labels: {no_pred}")
    print(
        f"Classification report for classifier :\n"
        f"{classification_report(y_real, predicted,labels= list(set(predicted)))}\n"
    )
    print("\n\n")


def flatten_img(images):
    n_samples = len(images)
    return images.reshape((n_samples, -1))


if __name__ == "__main__":
    main()
