# By Sergey Petrushkevich
# Wine Quality and Human Activity Recognition Classification

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# -------- WINE QUALITY DATASET --------

# Load the Wine Quality dataset
wine_data_red = pd.read_csv("wine+quality/winequality-red.csv", sep=";")
wine_data_white = pd.read_csv("wine+quality/winequality-white.csv", sep=";")
wine_data = pd.concat([wine_data_red, wine_data_white], ignore_index=True)

# Separate features and labels
X_wine = wine_data.iloc[:, :-1].values
y_wine = wine_data.iloc[:, -1].values

# Split the data into training and test sets
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42
)

# Calculate class priors, means, and covariance matrices
classes_wine = np.unique(y_train_wine)
class_priors_wine = {}
class_means_wine = {}
class_covariances_wine = {}

for c in classes_wine:
    X_c = X_train_wine[y_train_wine == c]
    class_priors_wine[c] = X_c.shape[0] / X_train_wine.shape[0]
    class_means_wine[c] = np.mean(X_c, axis=0)
    class_covariances_wine[c] = np.cov(X_c, rowvar=False) + 0.01 * np.eye(X_c.shape[1])


# Implement the classifier
def classify_samples_wine(X):
    posteriors = []
    for c in classes_wine:
        likelihood = multivariate_normal.pdf(
            X, mean=class_means_wine[c], cov=class_covariances_wine[c]
        )
        posterior = likelihood * class_priors_wine[c]
        posteriors.append(posterior)
    return classes_wine[np.argmax(posteriors, axis=0)]


# Classify the test samples and calculate the error rate
y_pred_wine = classify_samples_wine(X_test_wine)
error_rate_wine = np.mean(y_pred_wine != y_test_wine)

# Confusion matrix
confusion_matrix_wine = pd.crosstab(
    y_test_wine, y_pred_wine, rownames=["True"], colnames=["Predicted"], margins=True
)

print("Error rate (Wine Quality):", error_rate_wine)
print("Confusion Matrix (Wine Quality):")
print(confusion_matrix_wine)

# Visualize the dataset using PCA
pca = PCA(n_components=2)
X_pca_wine = pca.fit_transform(X_wine)

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=X_pca_wine[:, 0],
    y=X_pca_wine[:, 1],
    hue=y_wine,
    palette="viridis",
    s=70,
    alpha=0.7,
)
plt.title("Wine Quality Dataset - PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Wine Quality", loc="best", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("wine_quality_pca.png")
plt.show()

# -------- HUMAN ACTIVITY RECOGNITION DATASET --------

# Load the Human Activity Recognition dataset
X_train_har = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt"
)
y_train_har = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt"
).astype(int)
X_test_har = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/test/X_test.txt"
)
y_test_har = np.loadtxt(
    "human+activity+recognition+using+smartphones/UCI HAR Dataset/test/y_test.txt"
).astype(int)

# Calculate class priors, means, and covariance matrices
classes_har = np.unique(y_train_har)
class_priors_har = {}
class_means_har = {}
class_covariances_har = {}

for c in classes_har:
    X_c = X_train_har[y_train_har == c]
    class_priors_har[c] = X_c.shape[0] / X_train_har.shape[0]
    class_means_har[c] = np.mean(X_c, axis=0)
    class_covariances_har[c] = np.cov(X_c, rowvar=False) + 0.01 * np.eye(X_c.shape[1])


# Implement the classifier
def classify_samples_har(X):
    posteriors = []
    for c in classes_har:
        likelihood = multivariate_normal.pdf(
            X, mean=class_means_har[c], cov=class_covariances_har[c]
        )
        posterior = likelihood * class_priors_har[c]
        posteriors.append(posterior)
    return classes_har[np.argmax(posteriors, axis=0)]


# Classify the test samples and calculate the error rate
y_pred_har = classify_samples_har(X_test_har)
error_rate_har = np.mean(y_pred_har != y_test_har)

# Confusion matrix
confusion_matrix_har = pd.crosstab(
    y_test_har, y_pred_har, rownames=["True"], colnames=["Predicted"], margins=True
)

print("Error rate (Human Activity Recognition):", error_rate_har)
print("Confusion Matrix (Human Activity Recognition):")
print(confusion_matrix_har)

# Visualize the dataset using PCA
pca = PCA(n_components=2)
X_pca_har = pca.fit_transform(X_test_har)

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=X_pca_har[:, 0],
    y=X_pca_har[:, 1],
    hue=y_test_har,
    palette="viridis",
    s=70,
    alpha=0.7,
)
plt.title("Human Activity Recognition Dataset - PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Activity", loc="best", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("human_activity_recognition_pca.png")
plt.show()
