import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np
import seaborn as sns

def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    distancias = 0
    for i in range(len(a)):
        distancias += abs(a[i] - b[i]) ** p
    return distancias ** (1 / p)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if len(X_train) != len(y_train):
            raise ValueError('Length of X_train and y_train must be equal.')
        elif k <= 0 or p <= 0:
            raise ValueError('k and p must be positive integers.')
        else:
            self.k = k
            self.p = p
            self.x_train = X_train
            self.y_train = y_train


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predicciones = []
        for punto in X:
            distancias = self.compute_distances(punto)
            vecinos_cercanos = self.get_k_nearest_neighbors(distancias)
            knn_l = self.y_train[vecinos_cercanos]
            predicciones.append(self.most_common_label(knn_l))
        return np.array(predicciones)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        probabilidades = []
        for punto in X:
            distancias = self.compute_distances(punto)
            vecinos_cercanos = self.get_k_nearest_neighbors(distancias)
            knn_l = self.y_train[vecinos_cercanos]
            prob = np.bincount(knn_l, minlength=len(np.unique(self.y_train))) / self.k
            probabilidades.append(prob)
        return np.array(probabilidades)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        distancias = []
        for x in self.x_train:
            distancia = minkowski_distance(point, x, self.p)
            distancias.append(distancia)
        return np.array(distancias)

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        indices = np.argsort(distances)
        vecinos_cercanos = []
        for i in range(self.k):
            vecinos_cercanos.append(indices[i])
        return np.array(vecinos_cercanos)
    
    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        contador = {}
        for nivel in knn_labels:
            if nivel in contador:
                contador[nivel] += 1
            else:
                contador[nivel] = 1
        
        comun = None
        maximo = 0
        for nivel, count in contador.items():
            if count > maximo:
                comun = nivel
                maximo = count

        return comun

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"


def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    tp = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    tn = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
    fp = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    fn = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }

import numpy as np

import numpy as np

import numpy as np

def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_probs = np.array(y_probs)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins, right=False) - 1
    
    true_proportions = np.zeros(n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            true_proportions[i] = np.mean(y_true_mapped[bin_mask])
    
    return {"bin_centers": bin_centers, "true_proportions": true_proportions}


def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    positive_probs = y_probs[y_true_mapped == 1]
    negative_probs = y_probs[y_true_mapped == 0]

    plt.hist(positive_probs, bins=n_bins, alpha=0.5, label="Positive Class")
    plt.hist(negative_probs, bins=n_bins, alpha=0.5, label="Negative Class")
    plt.legend()
    plt.title("Probability Histograms")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.show()

    return {
        "array_passed_to_histogram_of_positive_class": positive_probs,
        "array_passed_to_histogram_of_negative_class": negative_probs,
    }

import numpy as np
import matplotlib.pyplot as plt

def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_probs = np.array(y_probs)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins, right=False) - 1
    
    true_proportions = np.zeros(n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            true_proportions[i] = np.mean(y_true_mapped[bin_mask])
    
    return {"bin_centers": bin_centers, "true_proportions": true_proportions}

def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.
    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_probs = np.array(y_probs)
    thresholds = np.linspace(0, 1, 11)
    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))

        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return {"fpr": np.array(fpr), "tpr": np.array(tpr)}
