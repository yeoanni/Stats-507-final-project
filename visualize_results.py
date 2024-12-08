import matplotlib.pyplot as plt

models = ["Transformer", "LSTM", "Moving Average"]
pr_auc_scores = [0.857, 0.057, 0.045]
roc_auc_scores = [0.976, 0.545, 0.497]

plt.figure(figsize=(12, 6))
plt.bar(models, pr_auc_scores, color="blue", alpha=0.7)
plt.title("Precision-Recall AUC Comparison")
plt.ylabel("PR-AUC")
plt.xlabel("Models")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(models, roc_auc_scores, color="green", alpha=0.7)
plt.title("ROC-AUC Comparison")
plt.ylabel("ROC-AUC")
plt.xlabel("Models")
plt.ylim(0, 1)
plt.show()
