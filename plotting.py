import numpy as np
from matplotlib import pyplot as plt

rfs = "(Retrain from Scratch)"
fmnist_label_map = {
    '0': 'T-shirt/top', '1': 'Trouser', '2': 'Pullover',
    '3': 'Dress', '4': 'Coat', '5': 'Sandal',
    '6': 'Shirt', '7': 'Sneaker', '8': 'Bag',
    '9': 'Ankle boot'
}


def plot_report_diff(report_before, report_after, dataset_name, case, retrain_from_scratch):
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

    if dataset_name.upper() == "Fashion-MNIST".upper():
        mapped_classes = [fmnist_label_map.get(c, c) for c in classes]
    else:
        mapped_classes = classes

    # --- Plot Precision ---
    precision_before = [report_before[c]['precision'] for c in classes]
    precision_after = [report_after[c]['precision'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, precision_before, width,
                    label=f'{"Before Unlearning" if not retrain_from_scratch else "Before Retraining from Scratch"}',
                    color='skyblue')
    bars2 = plt.bar(x + width / 2, precision_after, width,
                    label=f'{"After Unlearning" if not retrain_from_scratch else "After Retraining from Scratch"}',
                    color='lightcoral')

    plt.axhline(y=weighted_precision_before, color='skyblue', linestyle='--', linewidth=1,
                label='Weighted Precision Before')
    plt.axhline(y=weighted_precision_after, color='lightcoral', linestyle='--', linewidth=1,
                label='Weighted Precision After')

    # Annotate the horizontal lines with their values.
    plt.annotate(f"{weighted_precision_before:.4f}",
                 xy=(1.02, weighted_precision_before),
                 xycoords=('axes fraction', 'data'),
                 xytext=(3, 0),
                 textcoords="offset points",
                 color='skyblue',
                 va='center')
    plt.annotate(f"{weighted_precision_after:.4f}",
                 xy=(1.02, weighted_precision_after),
                 xycoords=('axes fraction', 'data'),
                 xytext=(3, 0),
                 textcoords="offset points",
                 color='lightcoral',
                 va='center')

    plt.xlabel("Class")
    plt.ylabel("Precision")
    plt.title(f"{dataset_name}: Precision per Class for {case} {rfs if retrain_from_scratch else ""}")
    plt.xticks(x, mapped_classes)
    plt.legend(loc='lower left')

    annotate_bars_v1(bars1)
    annotate_bars_v1(bars2)

    plt.tight_layout()

    if retrain_from_scratch:
        plt.savefig(f"./resources/{dataset_name}/{case}/no_shared_models/precision_diff_{case}.png")
    else:
        plt.savefig(f"./resources/{dataset_name}/{case}/precision_diff_{case}.png")
    plt.show(block=False)
    plt.close()

    # --- Plot Recall ---
    recall_before = [report_before[c]['recall'] for c in classes]
    recall_after = [report_after[c]['recall'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, recall_before, width,
                    label=f'{"Before Unlearning" if not retrain_from_scratch else "Before Retraining from Scratch"}',
                    color='skyblue')
    bars2 = plt.bar(x + width / 2, recall_after, width,
                    label=f'{"After Unlearning" if not retrain_from_scratch else "After Retraining from Scratch"}',
                    color='lightcoral')

    plt.axhline(y=weighted_recall_before, color='skyblue', linestyle='--', linewidth=1, label='Weighted Recall Before')
    plt.axhline(y=weighted_recall_after, color='lightcoral', linestyle='--', linewidth=1, label='Weighted Recall After')

    # Annotate the horizontal lines with their values.
    plt.annotate(f"{weighted_recall_before:.4f}",
                 xy=(1.02, weighted_recall_before),
                 xycoords=('axes fraction', 'data'),
                 xytext=(3, 0),
                 textcoords="offset points",
                 color='skyblue',
                 va='center')
    plt.annotate(f"{weighted_recall_after:.4f}",
                 xy=(1.02, weighted_recall_after),
                 xycoords=('axes fraction', 'data'),
                 xytext=(3, 0),
                 textcoords="offset points",
                 color='lightcoral',
                 va='center')

    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title(f"{dataset_name}: Recall per Class for {case} {rfs if retrain_from_scratch else ""}")
    plt.xticks(x, mapped_classes)
    plt.legend(loc='lower left')
    annotate_bars_v1(bars1)
    annotate_bars_v1(bars2)

    plt.tight_layout()

    if retrain_from_scratch:
        plt.savefig(f"./resources/{dataset_name}/{case}/no_shared_models/recall_diff_{case}.png")
    else:
        plt.savefig(f"./resources/{dataset_name}/{case}/recall_diff_{case}.png")
    plt.show(block=False)
    plt.close()

    # --- Plot F1-Score ---
    f1_before = [report_before[c]['f1-score'] for c in classes]
    f1_after = [report_after[c]['f1-score'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, f1_before, width,
                    label=f'{"Before Unlearning" if not retrain_from_scratch else "Before Retraining from Scratch"}',
                    color='skyblue')
    bars2 = plt.bar(x + width / 2, f1_after, width,
                    label=f'{"After Unlearning" if not retrain_from_scratch else "After Retraining from Scratch"}',
                    color='lightcoral')
    plt.axhline(y=weighted_f1_score_before, color='skyblue', linestyle='--', linewidth=1,
                label='Weighted F1-Score Before')
    plt.axhline(y=weighted_f1_score_after, color='lightcoral', linestyle='--', linewidth=1,
                label='Weighted F1-Score After')

    # Annotate the horizontal lines with their values.
    plt.annotate(f"{weighted_f1_score_before:.4f}",
                 xy=(1.02, weighted_f1_score_before),
                 xycoords=('axes fraction', 'data'),
                 xytext=(3, 0),
                 textcoords="offset points",
                 color='skyblue',
                 va='center')
    plt.annotate(f"{weighted_f1_score_after:.4f}",
                 xy=(1.02, weighted_f1_score_after),
                 xycoords=('axes fraction', 'data'),
                 xytext=(3, 0),
                 textcoords="offset points",
                 color='lightcoral',
                 va='center')
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title(f"{dataset_name}: F1-Score per Class for {case} {rfs if retrain_from_scratch else ""}")
    plt.xticks(x, mapped_classes)
    plt.legend(loc='lower left')

    annotate_bars_v1(bars1)
    annotate_bars_v1(bars2)

    plt.tight_layout()

    if retrain_from_scratch:
        plt.savefig(f"./resources/{dataset_name}/{case}/no_shared_models/f1_diff_{case}.png")
    else:
        plt.savefig(f"./resources/{dataset_name}/{case}/f1_diff_{case}.png")
    plt.show(block=False)
    plt.close()


def plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                             confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after, dataset_name, case,
                             retrain_from_scratch):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Bar width for before/after groups
    width = 0.35

    # 1. Loss on forgotten client's data
    ax = axs[0, 0]
    metrics = ['Loss']
    x = np.arange(len(metrics))
    bars_before = ax.bar(x - width / 2, [loss_before], width,
                         label=f'{"Before Unlearning" if not retrain_from_scratch else "Before Retraining from Scratch"}',
                         color='skyblue')
    bars_after = ax.bar(x + width / 2, [loss_after], width,
                        label=f'{"After Unlearning" if not retrain_from_scratch else "After Retraining from Scratch"}',
                        color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Loss')
    ax.set_title(f"{dataset_name}: Loss on Forgotten Client's Data for {case} {rfs if retrain_from_scratch else ""}")
    ax.legend(loc='lower left')
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
    bars_before = ax.bar(x - width / 2, [confi_forgotten_before], width,
                         label=f'{"Before Unlearning" if not retrain_from_scratch else "Before Retraining from Scratch"}',
                         color='skyblue')
    bars_after = ax.bar(x + width / 2, [confi_forgotten_after], width,
                        label=f'{"After Unlearning" if not retrain_from_scratch else "After Retraining from Scratch"}',
                        color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Confidence')
    ax.set_title(
        f"{dataset_name}: Confidence on Forgotten Client Data for {case} {rfs if retrain_from_scratch else ""}")
    ax.legend(loc='lower left')
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

    # 3. Confidence on unseen data
    ax = axs[1, 0]
    metrics = ['Conf. on Unseen']
    x = np.arange(len(metrics))
    bars_before = ax.bar(x - width / 2, [confi_unseen_before], width,
                         label=f'{"Before Unlearning" if not retrain_from_scratch else "Before Retraining from Scratch"}',
                         color='skyblue')
    bars_after = ax.bar(x + width / 2, [confi_unseen_after], width,
                        label=f'{"After Unlearning" if not retrain_from_scratch else "After Retraining from Scratch"}',
                        color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Confidence')
    ax.set_title(f"{dataset_name}: Confidence on Unseen Data for {case} {rfs if retrain_from_scratch else ""}")
    ax.legend(loc='lower left')
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

    # 4. Membership Inference Attack Accuracy
    ax = axs[1, 1]
    metrics = ['MIA Accuracy']
    x = np.arange(len(metrics))
    bars_before = ax.bar(x - width / 2, [mia_acc_before], width,
                         label=f'{"Before Unlearning" if not retrain_from_scratch else "Before Retraining from Scratch"}',
                         color='skyblue')
    bars_after = ax.bar(x + width / 2, [mia_acc_after], width,
                        label=f'{"After Unlearning" if not retrain_from_scratch else "After Retraining from Scratch"}',
                        color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Accuracy')
    ax.set_title(
        f"{dataset_name}: Membership Inference Attack Accuracy for {case} {rfs if retrain_from_scratch else ""}")

    ax.legend(loc='lower left')
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

    fig.tight_layout()

    if retrain_from_scratch:
        plt.savefig(f"./resources/{dataset_name}/{case}/no_shared_models/unlearning_findings_{case}.png")
    else:
        plt.savefig(f"./resources/{dataset_name}/{case}/unlearning_findings_{case}.png")
    plt.show(block=False)
    plt.close()


def plot_resource_comparison(cpu_usage_retrain, exec_time_retrain, cpu_usage_dfvu, exec_time_dfvu, dataset_name, case):
    methods = ['Retraining', 'DFVU-Unlearning']

    cpu_usages = [cpu_usage_retrain, cpu_usage_dfvu]
    exec_times = [exec_time_retrain, exec_time_dfvu]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(methods, cpu_usages, color=['skyblue', 'lightcoral'])
    axes[0].set_title(f'{dataset_name}: Total CPU Usage')
    axes[0].set_ylabel('CPU Usage (GB)')
    axes[0].set_ylim(0, max(cpu_usages) * 1.1)

    axes[1].bar(methods, exec_times, color=['skyblue', 'lightcoral'])
    axes[1].set_title('Total Execution Time')
    axes[1].set_ylabel('Execution Time (seconds)')
    axes[1].set_ylim(0, max(exec_times) * 1.1)

    plt.suptitle(
        f'{dataset_name}: Resource Usage Comparison: Retraining From Scratch (Baseline) vs DFVU-Framework \n in {case}')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f"./resources/{dataset_name}/{case}/resource_consumption_{case}.png")
    plt.show(block=False)
    plt.close()


def annotate_bars_v1(bar_container):
    for bar in bar_container:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')


# def plot_resource_comparison(cpu_usage_retrain, exec_time_retrain, cpu_usage_dfvu, exec_time_dfvu, dataset_name, case, retrain_from_scratch):
#     methods = ['Retraining', 'DFVU-Unlearning']
#
#     cpu_usages = [cpu_usage_retrain, cpu_usage_dfvu]
#     exec_times = [exec_time_retrain, exec_time_dfvu]
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
#     axes[0].bar(methods, cpu_usages, color=['skyblue', 'lightcoral'])
#     axes[0].set_title(f'{dataset_name}: Total CPU Usage')
#     axes[0].set_ylabel('CPU Usage (GB)')
#     axes[0].set_ylim(0, max(cpu_usages)*1.1)
#
#     axes[1].bar(methods, exec_times, color=['skyblue', 'lightcoral'])
#     axes[1].set_title('Total Execution Time')
#     axes[1].set_ylabel('Execution Time (seconds)')
#     axes[1].set_ylim(0, max(exec_times)*1.1)
#
#     plt.suptitle(f'{dataset_name}: Resource Usage Comparison: Retraining From Scratch (Baseline) vs DFVU-Framework \n in {case}')
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(f"./resources/{dataset_name}/{case}/no_shared_models/resource_consumption_{case}.png")
#     pltblock=False.show()
# plt.close()


def annotate_bars(bar_container):
    for bar in bar_container:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')


def plot_difference_retrain(f1_before_dvfu, f1_before_retrain, f1_after_dvfu, f1_after_retrain, dataset_name, case):
    # Calculate differences
    classes = [str(i) for i in range(10)]

    if dataset_name.upper() == "Fashion-MNIST".upper():
        mapped_classes = [fmnist_label_map.get(c, c) for c in classes]
    else:
        mapped_classes = classes

    diff_approach1 = np.array(f1_after_dvfu) - np.array(f1_before_dvfu)
    diff_approach2 = np.array(f1_after_retrain) - np.array(f1_before_retrain)

    x = np.arange(len(classes))
    width = 0.35
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, diff_approach1, width, label='DVFU Approach: Δ(After - Before)', color='skyblue')
    bars2 = plt.bar(x + width / 2, diff_approach2, width, label='Retrain from Scratch: Δ(After - Before)',
                    color='lightcoral')
    plt.ylabel('Difference in F1-Score: DVFU vs Retrain from Scratch')
    plt.xlabel('Class')
    plt.title(f'{dataset_name}: Difference in F1-Scores per Class: DVFU vs Retrain from Scratch for {case}')
    plt.xticks(x, mapped_classes)
    plt.legend(loc='lower left')

    # Annotate bar values
    def annotate_bars(bars):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom' if yval >= 0 else 'top',
                     ha='center')

    annotate_bars(bars1)
    annotate_bars(bars2)
    plt.tight_layout()
    plt.savefig(f"./resources/{dataset_name}/{case}/dvfu_retrain_f1_diff_{case}.png")

    plt.show(block=False)
    plt.close()


def resource_usage(cpu_mem_mnist, total_time_mnist, cpu_mem_fashion, total_time_fashion):
    labels = ['IID', 'IID RTFS', 'Non-IID', 'Non-IID RTFS', 'Extreme', 'Extreme RTFS']
    plt.figure()

    plt.bar(labels, cpu_mem_mnist)
    plt.title("CPU Memory Usage for MNIST")
    plt.xlabel("Scenario")
    plt.ylabel("CPU Memory (GB)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"./resources/MNIST/cpu_diff_with_rtfs.png")

    plt.show()

    # Plot Total Time for MNIST
    plt.figure()
    plt.bar(labels, total_time_mnist)
    plt.title("Total Training Time for MNIST")
    plt.xlabel("Scenario")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"./resources/MNIST/time_diff_with_rtfs.png")
    plt.show()

    # Plot CPU Memory Usage for Fashion-MNIST
    plt.figure()
    plt.bar(labels, cpu_mem_fashion)
    plt.title("CPU Memory Usage for Fashion-MNIST")
    plt.xlabel("Scenario")
    plt.ylabel("CPU Memory (GB)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"./resources/Fashion-MNIST/time_diff_with_rtfs.png")
    plt.show()

    # Plot Total Time for Fashion-MNIST
    plt.figure()
    plt.bar(labels, total_time_fashion)
    plt.title("Total Training Time for Fashion-MNIST")
    plt.xlabel("Scenario")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"./resources/Fashion-MNIST/time_diff_with_rtfs.png")
    plt.show()


if __name__ == "__main__":
    pass
    # MNIST - IID

    f1_before_dvfu = [0.96, 0.97, 0.96, 0.89, 0.96, 0.71, 0.96, 0.95, 0.86, 0.92]
    f1_after_dvfu = [0.93, 0.97, 0.95, 0.84, 0.95, 0.41, 0.96, 0.96, 0.86, 0.93]
    f1_before_retrain = [0.95, 0.98, 0.93, 0.88, 0.95, 0.62, 0.96, 0.94, 0.82, 0.92]
    f1_after_retrain = [0.95, 0.98, 0.96, 0.83, 0.96, 0.24, 0.94, 0.95, 0.84, 0.93]
    plot_difference_retrain(f1_before_dvfu, f1_before_retrain, f1_after_dvfu, f1_after_retrain, "MNIST", "IID-Case")

    # #MNIST - NON IID
    # f1_before_dvfu = [0.98, 0.98, 0.97, 0.97, 0.97, 0.97, 0.97, 0.96, 0.96, 0.95]
    # f1_after_dvfu = [0.96, 0.98, 0.94, 0.87, 0.94, 0.84, 0.96, 0.93, 0.91, 0.91]
    # f1_before_retrain = [0.98, 0.98, 0.97, 0.95, 0.96, 0.95, 0.96, 0.96, 0.94, 0.95]
    # f1_after_retrain = [0.95, 0.96, 0.88, 0.84, 0.90, 0.79, 0.92, 0.91, 0.85, 0.87]
    # plot_difference_retrain(f1_before_dvfu, f1_before_retrain, f1_after_dvfu, f1_after_retrain, "MNIST", "Non-IID-Case")
    # #
    # # #MNIST - EXTREME
    # f1_before_dvfu = [0.98, 0.99, 0.97, 0.96, 0.97, 0.97, 0.97, 0.97, 0.95, 0.96]
    # f1_after_dvfu = [0.93, 0.97, 0.80, 0.84, 0.91, 0.0, 0.94, 0.93, 0.78, 0.90]
    # f1_before_retrain = [0.98, 0.99, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.95, 0.95]
    # f1_after_retrain = [0.95, 0.95, 0.89, 0.84, 0.90, 0.78, 0.93, 0.91, 0.85, 0.87]
    # plot_difference_retrain(f1_before_dvfu, f1_before_retrain, f1_after_dvfu, f1_after_retrain, "MNIST", "Extreme-Non-IID-Case")
    #
    #
    #
    # #Fashion-MNIST - IID
    # f1_before_dvfu = [0.82, 0.97, 0.77, 0.87, 0.78, 0.93, 0.64, 0.92, 0.96, 0.94]
    # f1_after_dvfu = [0.8, 0.97, 0.75, 0.84, 0.77, 0.91, 0.59, 0.88, 0.94, 0.91]
    # f1_before_retrain = [0.82, 0.97, 0.76, 0.87, 0.7, 0.94, 0.65, 0.92, 0.97, 0.94]
    # f1_after_retrain = [0.81, 0.96, 0.73, 0.82, 0.76, 0.88, 0.60, 0.87, 0.92, 0.91]
    # plot_difference_retrain(f1_before_dvfu, f1_before_retrain, f1_after_dvfu, f1_after_retrain, "Fashion-MNIST", "IID-Case")
    #
    # #Fashion-MNIST - NON IID
    # f1_before_dvfu = [0.82, 0.97, 0.77, 0.87, 0.78, 0.94, 0.65, 0.91, 0.96, 0.94]
    # f1_after_dvfu = [0.8, 0.97, 0.75, 0.84, 0.76, 0.91, 0.63, 0.87, 0.93, 0.91]
    # f1_before_retrain = [0.81, 0.97, 0.76, 0.85, 0.78, 0.92, 0.65, 0.91, 0.96, 0.93]
    # f1_after_retrain = [0.80, 0.96, 0.74, 0.82, 0.75, 0.88, 0.54, 0.86, 0.91, 0.91]
    # plot_difference_retrain(f1_before_dvfu, f1_before_retrain, f1_after_dvfu, f1_after_retrain, "Fashion-MNIST", "Non-IID-Case")
    # #
    # # #Fashion-MNIST - EXTREME
    # f1_before_dvfu = [0.82, 0.97, 0.75, 0.87, 0.77, 0.93, 0.66, 0.92, 0.97, 0.94]
    # f1_after_dvfu = [0.81, 0.96, 0.74, 0.83, 0.76, 0.0, 0.58, 0.81, 0.69, 0.89]
    # f1_before_retrain = [0.82, 0.97, 0.76, 0.86, 0.77, 0.94, 0.63, 0.92, 0.96, 0.94]
    # f1_after_retrain = [0.80, 0.96, 0.73, 0.82, 0.76, 0.88, 0.60, 0.86, 0.92, 0.91]
    # plot_difference_retrain(f1_before_dvfu, f1_before_retrain, f1_after_dvfu, f1_after_retrain, "Fashion-MNIST", "Extreme-Non-IID-Case")
