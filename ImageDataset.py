########## Logistic Regression #####################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

df = pd.read_csv('dataset_full.csv')
print("Data successfully loaded!")

print(f"Dataset shape: {df.shape}")
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

print("Training the model")
classifier = LogisticRegression(
    random_state=0,
    max_iter=1000,
    multi_class='ovr'
)
classifier.fit(X_Train, Y_Train)


Y_Pred = classifier.predict(X_Test)

acc = accuracy_score(Y_Test, Y_Pred)
print(f"Model Accuracy: {acc * 100:.2f}%")


# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)
plt.figure(figsize=(8,6))
sns.heatmap(pd.DataFrame(cm), annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


n_classes = len(np.unique(Y))
Y_Test_bin = label_binarize(Y_Test, classes=np.arange(n_classes))

Y_Prob = classifier.predict_proba(X_Test)

plt.figure(figsize=(8,6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(Y_Test_bin[:, i], Y_Prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest Logistic Regression)')
plt.legend()
plt.grid()
plt.show()


########################### Kmeans ####################

from sklearn.cluster import KMeans

print("\n--- KMeans Clustering ---")


k = len(np.unique(Y_Train))
print(f"Number of classes detected: {k}")

kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
kmeans.fit(X_Train)

train_clusters = kmeans.labels_
cluster_map = {}

for i in range(k):

    indices_in_cluster = np.where(train_clusters == i)[0]

    if len(indices_in_cluster) > 0:
 
        true_labels = Y_Train[indices_in_cluster]
        values, counts = np.unique(true_labels, return_counts=True)
        most_common_label = values[np.argmax(counts)]

        cluster_map[i] = most_common_label
    else:
        
        cluster_map[i] = Y_Train[0]

print("Cluster Mapping Created (Cluster ID -> True Label):")
print(cluster_map)


test_clusters = kmeans.predict(X_Test)

Y_Pred_KMeans = np.array([cluster_map[c] for c in test_clusters])

kmeans_acc = accuracy_score(Y_Test, Y_Pred_KMeans)
print(f"KMeans Classification Accuracy: {kmeans_acc * 100:.2f}%")

cm_kmeans = confusion_matrix(Y_Test, Y_Pred_KMeans)

plt.figure(figsize=(8,6))
sns.heatmap(pd.DataFrame(cm_kmeans), annot=True, fmt='g', cmap='Greens')
plt.title('Confusion Matrix (KMeans Classifier)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()