import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    values_mean = [
        input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
        input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
        input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
        input_data['fractal_dimension_mean']
    ]

    values_se = [
        input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
        input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
        input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
    ]

    values_worst = [
        input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
        input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
        input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
        input_data['fractal_dimension_worst']
    ]

    # Close the circular plot
    values_mean += values_mean[:1]
    values_se += values_se[:1]
    values_worst += values_worst[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values_mean, color='blue', linewidth=2, linestyle='solid', label='Mean Value')
    ax.fill(angles, values_mean, color='blue', alpha=0.2)

    ax.plot(angles, values_se, color='green', linewidth=2, linestyle='dashed', label='Standard Error')
    ax.fill(angles, values_se, color='green', alpha=0.2)

    ax.plot(angles, values_worst, color='red', linewidth=2, linestyle='dotted', label='Worst Value')
    ax.fill(angles, values_worst, color='red', alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels([])

    ax.legend(loc='upper right')

    return fig


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = get_clean_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
        
    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    
    return scaled_dict


def load_model():
    """Load the individual models and create a soft voting ensemble"""
    try:
        # Try to load the pre-saved ensemble model
        ensemble = pickle.load(open("model/ensemble_model.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        return ensemble, scaler
    except (FileNotFoundError, IOError):
        # If the ensemble model doesn't exist, we'll create it
        st.warning("Ensemble model not found. Creating a new ensemble model with default parameters.")
        
        # Create base models with probability estimates enabled
        logistic = LogisticRegression(random_state=42)
        svc = SVC(probability=True, random_state=42)
        decision_tree = DecisionTreeClassifier(random_state=42)
        random_forest = RandomForestClassifier(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        
        # Create and return the Voting Classifier with soft voting
        ensemble = VotingClassifier(
            estimators=[
                ('logistic', logistic),
                ('svc', svc),
                ('dt', decision_tree),
                ('rf', random_forest),
                ('knn', knn)
            ],
            voting='soft'
        )
        
        # For simplicity, we'll use a default scaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        # Note: In a real scenario, you would train this ensemble on your dataset
        # ensemble.fit(X_train, y_train)
        # pickle.dump(ensemble, open("model/ensemble_model.pkl", "wb"))
        # pickle.dump(scaler, open("model/scaler.pkl", "wb"))
        
        return ensemble, scaler


def add_predictions(input_data):
    ensemble, scaler = load_model()
    
    # Extract values as array and reshape for prediction
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    # Scale the input
    input_array_scaled = scaler.transform(input_array)
    
    # Get model predictions
    prediction = ensemble.predict(input_array_scaled)
    
    # Display the prediction
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
    # For soft voting ensemble we want to show individual model predictions
    try:
        # Try to get predict_proba from the ensemble
        probabilities = ensemble.predict_proba(input_array_scaled)[0]
        st.write("Probability of being benign: ", probabilities[0])
        st.write("Probability of being malicious: ", probabilities[1])
        
        # Show individual model predictions if available
        st.subheader("Individual Model Predictions")
        
        try:
            for name, model in ensemble.named_estimators_.items():
                model_proba = model.predict_proba(input_array_scaled)[0]
                benign_prob = model_proba[0]
                malignant_prob = model_proba[1]
                
                # Create a horizontal bar chart for this model's probabilities
                fig, ax = plt.subplots(figsize=(8, 1))
                bars = ax.barh([''], [benign_prob], color='green', label='Benign')
                bars = ax.barh([''], [malignant_prob], left=[benign_prob], color='red', label='Malignant')
                
                ax.set_xlim(0, 1)
                ax.set_title(f"{name.capitalize()} Model")
                ax.legend(loc='upper right', bbox_to_anchor=(1, 1.5), ncol=2)
                
                st.pyplot(fig)
                
        except AttributeError:
            st.write("Individual model predictions unavailable.")
    
    except AttributeError:
        st.write("Probability estimates unavailable.")
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def show_model_comparison():
    """Display a comparison of model performance metrics (if available)"""
    st.subheader("Model Performance Comparison")
    
    try:
        # Try to load precomputed model metrics
        metrics = pickle.load(open("model/model_metrics.pkl", "rb"))
        
        # Create a bar chart comparing model accuracies
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(metrics.keys())
        accuracies = [metrics[model]['accuracy'] for model in models]
        
        bars = ax.bar(models, accuracies, color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        
        # Add accuracy values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Show a table with more detailed metrics
        metrics_df = pd.DataFrame({
            'Model': models,
            'Accuracy': [metrics[model]['accuracy'] for model in models],
            'Precision': [metrics[model].get('precision', 'N/A') for model in models],
            'Recall': [metrics[model].get('recall', 'N/A') for model in models],
            'F1 Score': [metrics[model].get('f1', 'N/A') for model in models]
        })
        
        st.table(metrics_df)
        
    except (FileNotFoundError, IOError):
        st.info("Model comparison metrics not available. Train and evaluate models to see comparative performance.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a soft voting ensemble of machine learning models whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")
    
    # Add a tabs interface
    tab1, tab2, tab3 = st.tabs(["Prediction", "Visualization", "Model Comparison"])
    
    with tab1:
        add_predictions(input_data)
    
    with tab2:
        radar_chart = get_radar_chart(input_data)
        st.pyplot(radar_chart)
        
    with tab3:
        show_model_comparison()


if __name__ == '__main__':
    main()
