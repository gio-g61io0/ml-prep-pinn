import GallenModel_v1
import numpy as np
import sklearn
from matplotlib import path, pyplot as plt
import matplotlib.colors as mcolors
import contextily as cx
import seaborn as sns
from torch import norm
from data import dataframe_to_dataset
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_confusion_matrix(preds, test_y):
    y_pred_classes = (preds > 0.5).astype("int32")
    cm = confusion_matrix(test_y, y_pred_classes)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
def find_best_threshold(y_true, y_pred_probs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh, fpr, tpr


def plot_distribution(df, title, x_label, y_label, label):
    #Plotting the distribution of bulk unit mean
    ax =sns.histplot(df[label], bins=30, kde=True, color="red")
    plt.title(title)
    # plt.axvline(np.mean(df[label]), color=)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_predicted_observed_map(gdf, predicted_col, observed_col):
    fig, axs = plt.subplots(1, 2, dpi=300, figsize=(8, 7))
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)

    cbar = fig.colorbar(sm, ax=axs[0])
    cbar.set_ticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    cbar.set_ticklabels(["0.0", "0.125", "0.375", "0.625", "0.875", "1.0"])

    gdf.plot(column=observed_col, cmap='RdYlGn_r', norm=norm, ax=axs[0])
    cx.add_basemap(axs[0], source=cx.providers.CartoDB.Positron)
    axs[0].set_title("Observed/Landslide Inventory")
    axs[0].set_axis_off()



    cbar = fig.colorbar(sm, ax=axs[1])
    cbar.set_ticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    cbar.set_ticklabels(["0.0", "0.125", "0.375", "0.625", "0.875", "1.0"])

    gdf.plot(column=predicted_col, cmap='RdYlGn_r', norm=norm, ax=axs[1],legend_kwds={"label": "Predicted Susceptibility Map"},)
    cx.add_basemap(axs[1], source=cx.providers.CartoDB.Positron)
    axs[1].set_title("Predicted Susceptibility")
    axs[1].set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_susceptibility_map(gdf, predictions, label_name, title="PINN Susceptibility Map"):
    gdf[f'predicted_susceptibility'] = predictions
    norm = mcolors.Normalize(vmin=0, vmax=1.0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Remove legend=True here
    gdf.plot(column=f'predicted_susceptibility', cmap='plasma_r', ax=ax, norm=norm)
    # gdf.plot(column=f'Landslide', cmap='plasma_r', ax=ax, norm=norm)

    # Add single custom colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm)
    # sm._A = []  
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    cbar.set_ticklabels(["0.0", "0.125", "0.375", "0.625", "0.875", "1.0"])

    ax.set_title(f"{title} - {label_name}")
    cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
    plt.tight_layout()
    plt.show()

def bootstrap_geotech(df, model, columns,filepath, n_bootstrap=50, ):

    data = dataframe_to_dataset(df[columns], shuffle=False)
    geotech_model_cohesion = tf.keras.Model(inputs=model.input, outputs=(model.get_layer("cohesion_layer").output * 5))
    geotech_model_ifi = tf.keras.Model(inputs=model.input, outputs=(model.get_layer("internal_friction").output))
    for i in range(1, n_bootstrap + 1):
        cohesion = geotech_model_cohesion.predict(data)
        ifi = geotech_model_ifi.predict(data)
        np.save(f"{filepath}/cohesion_bootstrap_{i}.npy", cohesion)
        np.save(f"{filepath}/ifi_bootstrap_{i}.npy", ifi)



def roc_auc_score_multiclass(actual_class, pred_class, average='macro'):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    
    print(f"Pred class: {pred_class}")
    for per_class in unique_class:
        y_true = (actual_class == per_class).astype(int)

        y_score = pred_class[:, per_class]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
        auc = sklearn.metrics.auc(fpr, tpr)
        roc_auc_dict[per_class] = [fpr, tpr, auc]

    return roc_auc_dict


def plot_auc(y_true, y_pred_probs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    auc = sklearn.metrics.auc(fpr, tpr)
    acc = round(sklearn.metrics.balanced_accuracy_score(y_true, y_pred_probs > 0.5), 2)

    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.plot(fpr, tpr, color="blue", label=f"(AUC={auc:.2f}, Accuracy={acc})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def plot_auc_with_distribution(y_true, y_pred_probs, bins=20):
    # ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot ROC curve on left y-axis
    
    ax1.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC={auc_score:.2f})")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("ROC Curve & Prediction Distribution")
    ax1.legend(loc="lower right")
    
    # Create a second y-axis for the histogram
    ax2 = ax1.twinx()
    
    # Histogram / bar plot of predicted probabilities
    ax2.hist(y_pred_probs, bins=bins, color="orange", alpha=0.3, label="Predicted Probabilities")
    ax2.set_ylabel("Count", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    
    # Optional: add legend for histogram
    ax2.legend(loc="upper center")
    
    plt.show()


def plot_auc_with_boxplot(y_true, y_pred_probs):
    # ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred_probs)
    auc_score = sklearn.metrics.auc(fpr, tpr)
    
    # Main figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC={auc_score:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve with Prediction Distribution")
    ax.legend(loc="lower right")
    
    # Inset axes for boxplot
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper left")  # adjust as needed
    ax_inset.boxplot(y_pred_probs, vert=True)
    ax_inset.set_title("Prediction Susceptibility", fontsize=10)
    ax_inset.set_ylabel("Predicted Prob", fontsize=8)
    ax_inset.tick_params(axis='x', labelbottom=False)  # hide x-axis ticks
    
    plt.show()

def plot_landslide_distribution(data):
    plt.bar(data.index, data.values, color=["skyblue", "salmon"])
    plt.xticks([0, 1], ["Non-Landslide (0)", "Landslide (1)"])
    plt.ylabel("Count")
    plt.title("Distribution of Landslide vs Non-Landslide")
    plt.show()


def calculate_distribution(df, column = 'predicted_susceptibility'):
    ranges = [[0, 0.125], [0.125, 0.375], [0.0375, 0.625], [0.625, 0.875], [0.875, 1.0]]
    
    range_values = {
        "0.125":0,
        "0.375":0,
        "0.625":0,
        "0.875":0,
        "1.0":0
    }
    for range in ranges:
        count = df[(df[column] > range[0]) & (df[column] < range[1])].shape[0]
        range_values[str(range[1])] = count
        
    return range_values

class OrdinalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="ordinal_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.acc = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_cls = tf.reduce_sum(y_true, axis=1)
        y_pred_cls = tf.reduce_sum(tf.cast(y_pred > 0.5, tf.float32), axis=1)
        self.acc.update_state(y_true_cls, y_pred_cls)

    def result(self):
        return self.acc.result()

    def reset_states(self):
        self.acc.reset_states()
