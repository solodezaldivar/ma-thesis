from matplotlib import pyplot as plt
import numpy as np

def plot_report_diff(report_before, report_after, dataset_name, case):
    exclude_keys = ["accuracy", "macro avg", "weighted avg"]

    acc_before = report_before.get("accuracy", 0) * 100
    acc_after = report_after.get("accuracy", 0) * 100

    print("acc_before", acc_before)
    print("acc_after", acc_after)

    weighted_avg_before = report_before.get("weighted avg", 0)
    weighted_precision_before = weighted_avg_before['precision']
    weighted_recall_before = weighted_avg_before['recall']
    weighted_f1_score_before = weighted_avg_before['f1-score']

    weighted_avg_after = report_after.get("weighted avg", 0)
    weighted_precision_after = weighted_avg_after['precision']
    weighted_recall_after = weighted_avg_after['recall']
    weighted_f1_score_after = weighted_avg_after['f1-score']

    classes = [label for label in report_before.keys() if label not in exclude_keys]

    try:
        classes = sorted(classes, key=lambda x: int(x))
    except:
        classes = sorted(classes)

    n_classes = len(classes)
    x = np.arange(n_classes)
    width = 0.35

    # --- Plot Precision ---
    precision_before = [report_before[c]['precision'] for c in classes]
    precision_after  = [report_after[c]['precision'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, precision_before, width, label='Before', color='skyblue')
    bars2 = plt.bar(x + width/2, precision_after, width, label='After', color='lightcoral')

    plt.axhline(y=weighted_precision_before, color='skyblue', linestyle='--', linewidth=1, label='Weighted Precision Before')
    plt.axhline(y=weighted_precision_after, color='lightcoral', linestyle='--', linewidth=1, label='Weighted Precision After')

    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.title(f"{dataset_name}: Precision per Class for {case}")
    plt.xticks(x, classes)
    plt.legend()

    # Annotate bars with values
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"./resources/{dataset_name}/precision_diff_{case}.png")
    plt.show()


    # --- Plot Recall ---
    recall_before = [report_before[c]['recall'] for c in classes]
    recall_after  = [report_after[c]['recall'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, recall_before, width, label='Before', color='skyblue')
    bars2 = plt.bar(x + width/2, recall_after, width, label='After', color='lightcoral')

    plt.axhline(y=weighted_recall_before, color='skyblue', linestyle='--', linewidth=1, label='Weighted Recall Before')
    plt.axhline(y=weighted_recall_after, color='lightcoral', linestyle='--', linewidth=1, label='Weighted Recall After')


    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title(f"{dataset_name}: Recall per Class for {case}")
    plt.xticks(x, classes)
    plt.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"./resources/{dataset_name}/recall_diff_{case}.png")
    plt.show()


    # --- Plot F1-Score ---
    f1_before = [report_before[c]['f1-score'] for c in classes]
    f1_after  = [report_after[c]['f1-score'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, f1_before, width, label='Before', color='skyblue')
    bars2 = plt.bar(x + width/2, f1_after, width, label='After', color='lightcoral')
    plt.axhline(y=weighted_f1_score_before, color='skyblue', linestyle='--', linewidth=1, label='Weighted F1-Score Before')
    plt.axhline(y=weighted_f1_score_after, color='lightcoral', linestyle='--', linewidth=1, label='Weighted F1-Score After')
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title(f"{dataset_name}: F1-Score per Class for {case}")
    plt.xticks(x, classes)
    plt.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"./resources/{dataset_name}/f1_diff_{case}.png")
    plt.show()



def plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                             confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after, dataset_name, case):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Bar width for before/after groups
    width = 0.35

    # 1. Loss on forgotten client's data
    ax = axs[0, 0]
    metrics = ['Loss']
    x = np.arange(len(metrics))
    bars_before = ax.bar(x - width/2, [loss_before], width, label='Before', color='skyblue')
    bars_after  = ax.bar(x + width/2, [loss_after],  width, label='After', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Loss')
    ax.set_title(f"{dataset_name}: Loss on Forgotten Client's Data for {case}")
    ax.legend()
    for bar in bars_before:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars_after:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    # 2. Confidence on forgotten client's data
    ax = axs[0, 1]
    metrics = ['Conf. on Forgotten']
    x = np.arange(len(metrics))
    bars_before = ax.bar(x - width/2, [confi_forgotten_before], width, label='Before', color='skyblue')
    bars_after  = ax.bar(x + width/2, [confi_forgotten_after], width, label='After', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Confidence')
    ax.set_title(f"{dataset_name}: Confidence on Forgotten Client Data for {case}")
    ax.legend()
    for bar in bars_before:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars_after:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    # 3. Confidence on unseen data
    ax = axs[1, 0]
    metrics = ['Conf. on Unseen']
    x = np.arange(len(metrics))
    bars_before = ax.bar(x - width/2, [confi_unseen_before], width, label='Before', color='skyblue')
    bars_after  = ax.bar(x + width/2, [confi_unseen_after], width, label='After', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Confidence')
    ax.set_title(f"{dataset_name}: Confidence on Unseen Data for {case}")
    ax.legend()
    for bar in bars_before:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars_after:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    # 4. Membership Inference Attack Accuracy
    ax = axs[1, 1]
    metrics = ['MIA Accuracy']
    x = np.arange(len(metrics))
    bars_before = ax.bar(x - width/2, [mia_acc_before], width, label='Before', color='skyblue')
    bars_after  = ax.bar(x + width/2, [mia_acc_after], width, label='After', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Accuracy')
    ax.set_title(f"{dataset_name}: Membership Inference Attack Accuracy for {case}")

    ax.legend()
    for bar in bars_before:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars_after:
        ax.annotate(f'{bar.get_height():.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(f"./resources/{dataset_name}/unlearning_findings_{case}.png")
    plt.show()

def plot_resource_comparison(cpu_usage_retrain, exec_time_retrain, cpu_usage_dfvu, exec_time_dfvu, dataset_name, case):
    methods = ['Retraining', 'DFVU-Unlearning']

    cpu_usages = [cpu_usage_retrain, cpu_usage_dfvu]
    exec_times = [exec_time_retrain, exec_time_dfvu]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(methods, cpu_usages, color=['skyblue', 'lightcoral'])
    axes[0].set_title(f'{dataset_name}: Total CPU Usage')
    axes[0].set_ylabel('CPU Usage (GB)')
    axes[0].set_ylim(0, max(cpu_usages)*1.1)

    axes[1].bar(methods, exec_times, color=['skyblue', 'lightcoral'])
    axes[1].set_title('Total Execution Time')
    axes[1].set_ylabel('Execution Time (seconds)')
    axes[1].set_ylim(0, max(exec_times)*1.1)

    plt.suptitle(f'{dataset_name}: Resource Usage Comparison: Retraining From Scratch (Baseline) vs DFVU-Framework \n in {case}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"./resources/{dataset_name}/resource_consumption_{case}.png")
    plt.show()
