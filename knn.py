import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib
from sklearn.preprocessing import StandardScaler
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("df_cleaned_83.csv")
categorical_cols = ['Species', 'Population', 'Thorax_length', 'wing_loading','Sex']

X = df.drop(columns= categorical_cols)
y = df["Sex"]
# split into train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
train_accuracies = []
test_accuracies = []
def plot_knn_accuracies_original(k_range):
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))
        max_accuracy = np.max(test_accuracies)
        print(f"normal best test accuracy without StandardScaler = {max_accuracy} when k = 5 ")
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(k_range, test_accuracies, label='Test Accuracy', marker='o')
    plt.title('KNN Accuracy vs. K Value on Original Data')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    # # Find the highest test accuracy and the corresponding k value
    #
    # best_k = k_range[np.argmax(test_accuracies)]
    #
    # return best_k, max_accuracy



def plot_knn_accuracies_scaled(k_range):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_train_pred = knn.predict(X_train_scaled)
        y_test_pred = knn.predict(X_test_scaled)
        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(k_range, test_accuracies, label='Test Accuracy', marker='o')
    plt.title('KNN Accuracy vs. K Value on Scaled Data')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Find the highest test accuracy and the corresponding k value
    max_accuracy = np.max(test_accuracies)
    best_k = k_range[np.argmax(test_accuracies)]

    return best_k, max_accuracy

k_values = range(4,7)
best_k, max_accuracy = plot_knn_accuracies_original(k_values)
