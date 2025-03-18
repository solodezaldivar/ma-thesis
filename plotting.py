from matplotlib import pyplot as plt
import numpy as np

def plot_report_diff(report_before, report_after, dataset_name):
    exclude_keys = ["accuracy", "macro avg", "weighted avg"]
    
    acc_before = report_before.get("accuracy", 0) * 100
    acc_after = report_after.get("accuracy", 0) * 100
    
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
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Precision per Class')
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
    plt.savefig(f"./resources/{dataset_name}/precision_diff.png")
    plt.show()
    

    # --- Plot Recall ---
    recall_before = [report_before[c]['recall'] for c in classes]
    recall_after  = [report_after[c]['recall'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, recall_before, width, label='Before', color='skyblue')
    bars2 = plt.bar(x + width/2, recall_after, width, label='After', color='lightcoral')
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.title('Recall per Class')
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
    plt.savefig(f"./resources/{dataset_name}/recall_diff.png")
    plt.show()
    

    # --- Plot F1-Score ---
    f1_before = [report_before[c]['f1-score'] for c in classes]
    f1_after  = [report_after[c]['f1-score'] for c in classes]

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, f1_before, width, label='Before', color='skyblue')
    bars2 = plt.bar(x + width/2, f1_after, width, label='After', color='lightcoral')
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title('F1-Score per Class')
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
    plt.savefig(f"./resources/{dataset_name}/f1_diff.png")
    plt.show()



def plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after, confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after):
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
    ax.set_title("Loss on Forgotten Client's Data")
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
    ax.set_title("Confidence on Forgotten Client Data")
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
    ax.set_title("Confidence on Unseen Data")
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
    ax.set_title("Membership Inference Attack Accuracy")
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
    plt.savefig(f"./resources/MNIST/unlearning_findings.png")
    plt.show()
