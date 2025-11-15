from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB





def NB_eval(X_train, y_train, X_test):

    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = nb_classifier.predict(X_test)
    return y_pred

def MLP_eval(X_train, y_train, X_test):
    #mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1000)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu')

    mlp_classifier.fit(X_train, y_train)

    y_pred = mlp_classifier.predict(X_test)
    return y_pred

def KNN_eval(X_train, y_train, X_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)

    y_pred = knn_classifier.predict(X_test)
    return y_pred

def RF_eval(X_train, y_train, X_test):
    rf_classifier = RandomForestClassifier(n_estimators=10, criterion='gini')  # 设置随机森林分类器，这里选择了100棵决策树
    rf_classifier.fit(X_train, y_train)  # 在训练数据上训练随机森林模型

    y_pred = rf_classifier.predict(X_test)  # 使用训练好的模型对测试数据进行预测
    return y_pred

def CART_eval(X_train, y_train, X_test):
    cart_classifier = DecisionTreeClassifier()  # 创建一个 CART 分类器对象
    cart_classifier.fit(X_train, y_train)  # 在训练数据上训练 CART 模型

    y_pred = cart_classifier.predict(X_test)  # 使用训练好的模型对测试数据进行预测
    return y_pred

def SVM_eval(X_train, y_train, X_test):

    svm_classifier = SVC()  # 创建一个SVM分类器对象
    svm_classifier.fit(X_train, y_train)  # 在标准化的训练数据上训练SVM模型

    y_pred = svm_classifier.predict(X_test)  # 使用训练好的模型对标准化的测试数据进行预测
    return y_pred