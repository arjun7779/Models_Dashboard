import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import seaborn as sns
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import label_binarize
import ast  # Import ast to parse string lists

# Display Logo at the top with a white background
logo_path = "logo.jpg"  # Replace with your logo file path
logo = Image.open(logo_path)
st.image(logo, width=200, use_container_width=True, caption="", output_format="PNG", clamp=True)

# Title and Introduction
st.title("📊 Models Result and Analysis Dashboard")
st.write("This interactive dashboard showcases results from trained Machine Learning models.")

# Sidebar for user selections
with st.sidebar:
    st.header("🔧 Model Selection")
    model_choice = st.selectbox("Select model...",
                                ["Logistic Regression", "Random Forrest", "Support Vector Classifier"])
    parameters_choice = st.selectbox("Parameters Type...", ["Default Parameters", "Fine Tuned Parameters"])
    predictor_choice = st.selectbox("Predictor Variable...", ["Brake Duration", "Brake Intensity"])
    clustering_choice = st.selectbox("Clustering Type...",
                                     ["Kmeans", "Gaussian Mixture Models", "Agglomerative Hierarchical Clustering"])


# Load pre-saved model results from CSV files in 'Models' folder
@st.cache_data
def load_data(parameters_choice):
    base_path = "Models"  # Folder where model CSV files are stored
    files = {}

    if parameters_choice == "Default Parameters":
        files["Logistic Regression - Kmeans - Brake Duration"] = os.path.join(base_path,
                                                                              "log_reg_default_kmeans_brake_duration.csv")
        files["Logistic Regression - Kmeans - Brake Intensity"] = os.path.join(base_path,
                                                                               "log_reg_default_kmeans_brake_intensity.csv")
    else:  # Fine Tuned Parameters
        model_types = {"Logistic Regression": "log_reg", "Random Forrest": "random_forest", "Support Vector Classifier": "svc"}
        clustering_types = {"Kmeans": "kmeans", "Gaussian Mixture Models": "gmm", "Agglomerative Hierarchical Clustering": "agg"}
        predictors = {"Brake Duration": "brake_duration", "Brake Intensity": "brake_intensity"}

        for model_name, model_prefix in model_types.items():
            for cluster_name, cluster_prefix in clustering_types.items():
                for predictor_name, predictor_prefix in predictors.items():
                    filename = f"{model_prefix}_finetuned_{cluster_prefix}_{predictor_prefix}.csv"
                    key_name = f"{model_name} - {cluster_name} - {predictor_name}"
                    files[key_name] = os.path.join(base_path, filename)

    all_models_data = []
    for model_name, file_path in files.items():
        if os.path.exists(file_path):  # Ensure file exists before reading
            df = pd.read_csv(file_path)
            df['model_config'] = model_name  # Ensure unique model config is in the data
            all_models_data.append(df)
        else:
            st.warning(f"⚠️ Missing file: {file_path}")

    return pd.concat(all_models_data, ignore_index=True) if all_models_data else pd.DataFrame()


# Load data once and store in memory
df = load_data(parameters_choice)
# st.write("### Dataset Overview")
# st.dataframe(df)

# Filter dataframe based on selected model configuration
model_config = f"{model_choice} - {clustering_choice} - {predictor_choice}"
df_filtered = df[df['model_config'] == model_config]

# # Debugging info
# st.write("**🔍 Debugging Information**")
# st.write(f"Selected Model Config: `{model_config}`")
# st.write("Available Model Configs in Dataset:")
# st.write(df['model_config'].unique())

# Select Target and Predictions
target_col = predictor_choice
pred_col = 'prediction'  # Assuming a generic prediction column, adjust if needed


if not df_filtered.empty:
    y_true = df_filtered[target_col].astype(int)  # Ensure integer type
    y_pred = df_filtered[pred_col].astype(int)  # Ensure integer type

    # Select correct probability columns based on class count

    if predictor_choice == "Brake Duration":
        y_probs = df_filtered.filter(like='probability', axis=1).astype(float)
        y_probs = np.vstack(y_probs.to_numpy())

    else:
        num_classes = len(np.unique(y_true))
        probability_cols = [f'probability_{i}' for i in range(num_classes) if f'probability_{i}' in df_filtered.columns]
        y_probs = df_filtered[probability_cols].astype(float).to_numpy()

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    st.metric(label="### 📊 Model Accuracy", value=f"{accuracy*100}%")

    # Classification Report
    st.write("### 📑 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.write("### 🔲 Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve and AUC Score
    st.write("### 📉 Receiver Operating Characteristic (ROC) Curve")

    if y_probs.size > 0:
        if predictor_choice == "Brake Duration":  # Binary Classification
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 0])
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Model")
            ax.set_xlabel('False Positive Rate (FPR)')
            ax.set_ylabel('True Positive Rate (TPR)')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend()
            st.pyplot(fig)
        else:  # Multi-class Classification
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            fig, ax = plt.subplots()
            for i in range(y_true_bin.shape[1]):
                if i < y_probs.shape[1]:  # Ensure we don't exceed probability columns
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Multi-Class ROC Curve')
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Probability scores not available for ROC computation.")
