import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_summary_chart(results, reports_dir, timestamp, model_name, class_labels):
    """
    Generates a generic, combined chart with performance metrics and a confusion matrix.
    
    Args:
        results (dict): The dictionary containing evaluation metrics.
        reports_dir (str): The directory to save the report image.
        timestamp (str): The timestamp for the filename.
        model_name (str): The name of the model to be used in the chart title.
        class_labels (list): The labels for the confusion matrix axes (e.g., ['Benign', 'Malignant']).
    """
    print("\nGenerating combined performance summary chart...")
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [3, 2]})
    fig.suptitle(f'{model_name} Performance (Averaged over 50 Runs)', fontsize=14, weight='bold')

    # --- Plot 1: Bar Chart of Performance Metrics ---
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]

    metric_labels = [m.replace('_', ' ').capitalize() for m in metrics]
    bars = ax1.bar(metric_labels, means, yerr=stds, capsize=5, color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'], alpha=0.8)
    ax1.set_title('Key Performance Metrics (Mean ± STD)', fontsize=14)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(min(means) - max(stds) - 0.05, 1.0)

    for bar in bars:
        yval = bar.get_height()
        std = stds[bars.index(bar)]
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + std + 0.005, f'{yval:.4f} ± {std:.4f}', ha='center', va='bottom', fontsize=11)

    # --- Plot 2: Confusion Matrix Heatmap ---
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues', ax=ax2, cbar=False,
                xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 14})
    ax2.set_title('Average Confusion Matrix', fontsize=14)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    # --- Footnote Section ---
    avg_loss_mean = np.mean(results['best_losses'])
    avg_loss_std = np.std(results['best_losses'])
    avg_iter_mean = np.mean(results['iterations'])
    avg_iter_std = np.std(results['iterations'])
    best_run = results['best_run_index'] + 1
    best_acc = results['best_overall_accuracy']
    
    footnote_text = (
        f"Additional Results:\n"
        f"• Average Best Loss: {avg_loss_mean:.6f} ± {avg_loss_std:.6f}\n"
        f"• Average Iterations to Converge: {avg_iter_mean:.1f} ± {avg_iter_std:.1f}\n"
        f"• Best Performing Model: Run {best_run} with {best_acc:.2%} Accuracy"
    )
    plt.figtext(0.5, -0.05, footnote_text, ha="center", fontsize=12,
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    # --- Adjust layout and save the figure ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join(reports_dir, f'ACOR-LM_diabetes_perf_summary_{timestamp}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPerformance report saved to: {output_filename}")
    plt.show()
def generate_summary_chart(results, reports_dir, timestamp):
    """
    Generates a combined chart with performance metrics and a confusion matrix heatmap.
    """
    print("\nGenerating combined performance summary chart...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a figure with two subplots (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [3, 2]})
    fig.suptitle('ACOR-LM Cancer Model Performance (Averaged over 50 Runs)', fontsize=18, weight='bold')

    # --- Plot 1: Bar Chart of Performance Metrics ---
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    means = [np.mean(results[m]) for m in metrics]
    stds = [np.std(results[m]) for m in metrics]

    metric_labels = [m.replace('_', ' ').capitalize() for m in metrics]
    bars = ax1.bar(metric_labels, means, yerr=stds, capsize=5, color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'], alpha=0.8)
    ax1.set_title('Key Performance Metrics (Mean ± STD)', fontsize=14)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(min(means) - max(stds) - 0.05, 1.0)

    for bar in bars:
        yval = bar.get_height()
        std = stds[bars.index(bar)]
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + std + 0.005, f'{yval:.4f} ± {std:.4f}', ha='center', va='bottom', fontsize=11)

    # --- Plot 2: Confusion Matrix Heatmap ---
    avg_cm = np.mean(results['confusion_matrices'], axis=0)
    cm_labels = ['Benign (0)', 'Malignant (1)']
    
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues', ax=ax2, cbar=False,
                xticklabels=cm_labels, yticklabels=cm_labels, annot_kws={"size": 14})
    ax2.set_title('Average Confusion Matrix', fontsize=14)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    # --- Footnote Section ---
    avg_loss_mean = np.mean(results['best_losses'])
    avg_loss_std = np.std(results['best_losses'])
    avg_iter_mean = np.mean(results['iterations'])
    avg_iter_std = np.std(results['iterations'])
    best_run = results['best_run_index'] + 1
    best_acc = results['best_overall_accuracy']
    
    footnote_text = (
        f"Additional Results:\n"
        f"• Average Best Loss: {avg_loss_mean:.6f} ± {avg_loss_std:.6f}\n"
        f"• Average Iterations to Converge: {avg_iter_mean:.1f} ± {avg_iter_std:.1f}\n"
        f"• Best Performing Model: Run {best_run} with {best_acc:.2%} Accuracy"
    )
    plt.figtext(0.5, -0.05, footnote_text, ha="center", fontsize=12,
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})

    # --- Adjust layout and save the figure ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join(reports_dir, f'ACOR-LM_cancer_model_perf_summary_{timestamp}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPerformance report saved to: {output_filename}")
    plt.show()