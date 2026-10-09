import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def generate_craniodent_matrix(y_true, y_pred, save_path="craniodent_fusion_matrix.png"):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["Female", "Male"])
    acc = accuracy_score(y_true, y_pred)
    
    # Configure academic plot styling
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    
    # Create the heatmap
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=["Female", "Male"], 
                     yticklabels=["Female", "Male"],
                     annot_kws={"size": 16, "weight": "bold"},
                     cbar_kws={'label': 'Number of Subjects'},
                     linewidths=1, linecolor='black')
    
    # Customize labels and titles for the thesis
    plt.title(f'CranioDental Multimodal Consensus\nSex Classification (Accuracy: {acc*100:.1f}%)', 
              fontsize=14, pad=20, weight='bold')
    plt.ylabel('Actual Biological Sex', fontsize=12, weight='bold')
    plt.xlabel('Predicted Consensus Sex', fontsize=12, weight='bold')
    
    # Adjust layout and save high-res for LaTeX/Word
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Matrix saved successfully to {save_path}")
    plt.show()

if __name__ == '__main__':
    # ==========================================
    # TODO: Replace these lists with your actual 
    # test dataset results from the combined engine
    # ==========================================
    
    # Example: 100 test cases
    actual_labels = ["Female"] * 48 + ["Male"] * 52
    
    # Simulating a highly accurate consensus output (e.g., 98% accuracy)
    predicted_labels = (
        ["Female"] * 47 + ["Male"] * 1 +   # 1 False Positive for Female
        ["Male"] * 51 + ["Female"] * 1     # 1 False Negative for Male
    )
    
    generate_craniodent_matrix(actual_labels, predicted_labels)