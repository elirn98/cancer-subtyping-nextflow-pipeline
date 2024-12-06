from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import random
import time
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


# Arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--history_path', type=str, required=True, help='Path of trainHistoryDict')
parser.add_argument('--result_path', type=str, required=True, help='Path to test results')
parser.add_argument('--save_plot_path', type=str, required=True, help='Path to store ploted metrics')

args = parser.parse_args()
history_path = args.history_path
result_path = args.result_path
save_plot_path = args.save_plot_path

# Load statistics
with open(result_path, 'rb') as f:
    # Load the pickled data
    loaded_test_results = pickle.load(f)
    y_test =loaded_test_results['label_test']
    y_prediction = loaded_test_results['label_prediction']

with open(history_path, 'rb') as f:
    history = pickle.load(f)

# plot history for accuracy
# plt.plot(history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend('train', loc='upper left')
# plt.savefig(os.path.join(save_plot_path, 'model_accuracy.png'))
# plt.close()


# # plot history for loss
# plt.plot(history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend('train', loc='upper left')
# plt.savefig(os.path.join(save_plot_path, 'model_loss.png'))
# plt.close()

report = classification_report(y_test, y_prediction, labels=[2], target_names=["Class 2"])
print(report)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_prediction, labels=[0, 1, 2, 3, 4])

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CC', 'EC', 'HGSC', 'LGSC', 'MC'])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_plot_path, 'model_confusion_matrix.png'))
plt.close()

from sklearn.metrics import precision_recall_curve, average_precision_score

# Convert to binary labels for Class 2
binary_true_labels = (np.array(y_test) == 2).astype(int)
binary_predicted_labels = (np.array(y_prediction) == 2).astype(int)

# Precision-recall curve
precision, recall, _ = precision_recall_curve(binary_true_labels, binary_predicted_labels)
average_precision = average_precision_score(binary_true_labels, binary_predicted_labels)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'AP = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Class 2')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_plot_path, 'PR_curve.png'))
plt.close()
