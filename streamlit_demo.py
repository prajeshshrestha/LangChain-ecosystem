import streamlit as st
import numpy as np 

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

st.title("Streamlit example")

# markdown format
st.write("""
    # Exploring different functionalities of Streamlit
""")

# action trigger and recorder
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris Dataset", "Breast Cancer Dataset", "Wine Dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

# updating UI w.r.t. to the choosen dataset
def get_dataset(dataset_name):
    if dataset_name == 'Iris Dataset':
        data = datasets.load_iris()

    elif dataset_name == 'Breast Cancer Dataset':
        data = datasets.load_breast_cancer()

    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X, y

# constantly looking for the change and making corresponding update to the UI
X, y = get_dataset(dataset_name)
st.write("Shape of dataset is: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))

# callback function to the action triggers
# updating UI w.r.t. to the choosen classifier (params)
def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == 'KNN':
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators

    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    elif clf_name == "SVM":
        clf = SVC(C = params['C'])
    else:
        clf = RandomForestClassifier(n_estimators = params['n_estimators'], max_depth = params['max_depth'], random_state = 169)

    return clf

clf = get_classifier(classifier_name, params)

# classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"""
        # Classifier: {classifier_name}
        #### Accuracy: {round(acc * 100, 2)} %
""")

# lets have some plots
# feature reduction technique for 2d plots (PCA)
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c = y, alpha = 0.85, cmap = 'viridis')
plt.xlabel("Principle Components 1")
plt.ylabel("Principle Components 2")
plt.colorbar()

# replacing plt.show() with streamlit
st.pyplot(fig)