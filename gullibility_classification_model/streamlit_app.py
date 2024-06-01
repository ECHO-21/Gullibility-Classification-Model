# Import necessary libraries
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Function to load data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Separating numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Checking if there are numeric columns before applying imputation
    if not numeric_columns.empty:
        # Imputing missing values only for numeric columns
        imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    return df

# Function to visualize data
def visualize_data(df):
    st.subheader("Data Visualization")

    # Visualising the number of gullible and compliant users
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.countplot(x='Gullibility', data=df, ax=axes[0, 0])
    sns.boxplot(x='Gullibility', y='Age', data=df, ax=axes[0, 1])
    sns.boxplot(x='Gullibility', y='Extraversion', data=df, ax=axes[1, 0])
    sns.boxplot(x='Gullibility', y='Openness', data=df, ax=axes[1, 1])

    st.pyplot(fig)

    # Visualising the number of gullible and compliant users
    fig2, axes2 = plt.subplots()
    gullible_counts = df['Gullibility'].value_counts()
    labels = ['Compliant Users', 'Gullible (Deficient) Users']
    axes2.pie(gullible_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    axes2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.subheader("User Gullibility")
    st.pyplot(fig2)

    # Visualising the weekly security posture
    if 'Week' in df.columns and 'Security_Posture' in df.columns:
        # Visualising the weekly security posture
        fig, axes = plt.subplots(figsize=(10, 6))
        weekly_security_posture = df.groupby('Week')['Security_Posture'].mean().reset_index()
        sns.lineplot(x='Week', y='Security_Posture', data=weekly_security_posture, marker='o')
        plt.xlabel('Week')
        plt.ylabel('Security Posture')
        plt.title('Weekly Security Posture')

        st.subheader("Weekly Security Posture")
        st.pyplot(fig)
    else:
        st.write("Required columns not found in the DataFrame.")

    # Pair Plot for numerical variables
    st.subheader("Pair Plot")
    numerical_columns = ['Age', 'Extraversion', 'Openness', 'Conscientiousness', 'Curiosity']
    pair_plot = sns.pairplot(df, hue='Gullibility', vars=numerical_columns, palette='viridis')
    st.pyplot(pair_plot)
    st.write("Pair Plot showing relationships between numerical variables.")

# Function to train the model
def train_model(df):
    st.subheader("Model Training")


    X = df[['Age', 'Extraversion', 'Openness', 'Conscientiousness', 'Curiosity']]
    y = df['Gullibility']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training a Random Forest Classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    st.write("Model trained successfully.")
    return classifier

# Function to evaluate the classifier
def evaluate_classifier(classifier, df):
    st.subheader("Model Evaluation")

    X = df[['Age', 'Extraversion', 'Openness', 'Conscientiousness', 'Curiosity']]
    y = df['Gullibility']
    y_pred = classifier.predict(X)

    # Calculating classification report and confusion matrix
    report = classification_report(y, y_pred, output_dict=True)

    # Extracting scores and metrics
    metrics = {'Precision', 'Recall', 'F1-Score', 'Support'}
    class_labels = [str(label) for label in df['Gullibility'].unique()]  # Convert labels to strings

    # Displaying formatted classification report
    st.write("Classification Report:")
    st.write(f"{'':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")

    for label in class_labels:
        st.write(f"{label:<10} {report[label]['precision']:.2f} {report[label]['recall']:.2f} {report[label]['f1-score']:.2f} {report[label]['support']:.1f}")

    st.write(f"{'Accuracy':<10} {report['accuracy']:.2f}")

    # Displaying confusion matrix
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y, y_pred))

    # Displaying heatmap for the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    st.pyplot(fig)

    # Ploting ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1], '--', color='gray')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(['ROC Curve'])

    st.pyplot(fig_roc)

    # Ploting precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    ax_pr.plot(recall, precision)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(['Precision-Recall Curve'])

    st.pyplot(fig_pr)

# Function to predict gullibility
def predict_gullible(classifier, df, settings):
    st.subheader("Gullibility Prediction")

    # Allow user to select prediction type: individual or department
    prediction_type = st.radio("Select prediction type:", ("Individual", "Department"))

    if prediction_type == "Individual":
        predict_individual_gullibility(classifier, df, settings)
    else:
        predict_department_gullibility(classifier, df, settings)

# Function to predict individual gullibility
def predict_individual_gullibility(classifier, df, settings):
    st.subheader("Individual Gullibility Prediction")

    # Allow user to select an individual
    individual = st.selectbox("Select an individual:", df['Name'].unique())
    data = df[df['Name'] == individual]

    # Predict gullibility based on the model and settings
    prediction_probs = classifier.predict_proba(
        data[['Age', 'Extraversion', 'Openness', 'Conscientiousness', 'Curiosity']])[:, 1]
    prediction = (
            (prediction_probs > settings['gullibility_threshold']) &
            (data['Age'].between(*settings['age_range'])) &
            (data['Extraversion'] > settings['extraversion_threshold'])
    ).astype(int)

    # Confidence level chart
    fig_confidence = plt.figure(figsize=(8, 4))
    plt.barh(['Confidence Level'], [prediction_probs[0]], color=['blue'])
    plt.xlim(0, 1)
    plt.xlabel('Confidence Level')
    plt.title('Prediction Confidence Level')

    st.pyplot(fig_confidence)

    # Displaying prediction statement
    st.write(
        f"{individual} is{' ' if prediction.iloc[0] else ' not '}gullible with a confidence level of {prediction_probs[0]:.2f}.")

    # Displaying gullible individuals in a wide table
    st.subheader("Gullible Individuals:")
    st.dataframe(data[data['Gullibility'] == 1][['Name']], width=500, height=500)

# Function to predict department gullibility
def predict_department_gullibility(classifier, df, settings):
    st.subheader("Department Gullibility Prediction")

    # Grouping the data by department and calculate mean for relevant columns
    grouped_data = df.groupby('Department', as_index=False)[
        ['Age', 'Extraversion', 'Openness', 'Conscientiousness', 'Curiosity']].mean()

    # Predicting department gullibility based on the model and settings
    grouped_data['Gullibility_Prediction'] = (
            (classifier.predict_proba(
                grouped_data[['Age', 'Extraversion', 'Openness', 'Conscientiousness', 'Curiosity']])[:, 1] > settings[
                 'gullibility_threshold']) &
            (grouped_data['Age'].between(*settings['age_range'])) &
            (grouped_data['Extraversion'] > settings['extraversion_threshold'])
    ).astype(int)

    # Displaying gullible individuals and departments
    gullible_individuals = df[df['Gullibility'] == 1]['Name']
    gullible_departments = grouped_data[grouped_data['Gullibility_Prediction'] == 1]['Department']

    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Displaying gullible individuals in a wide table
    st.subheader("Gullible Individuals:")
    st.dataframe(df[df['Gullibility'] == 1][['Name']], width=500, height=500)

    # Displaying gullible departments in a wide table
    st.subheader("Gullible Departments:")
    st.dataframe(grouped_data[grouped_data['Gullibility_Prediction'] == 1][['Department']], width=500, height=500)

    # Reset pandas display options to default
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

# Function for the login page
def login():
    st.subheader("Login")

    actual_username = "admin"
    actual_password = "password"

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit = st.button("Login")

    if submit and username == actual_username and password == actual_password:
        st.session_state.login_status = True
        return True
    elif submit:
        st.error("Invalid username or password. Please try again.")
        st.session_state.login_status = False
    return False

# Function for the settings page
def settings():
    st.subheader("Settings")

    # Default values for settings
    default_settings = {
        'gullibility_threshold': 0.5,
        'age_range': (20, 40),
        'extraversion_threshold': 0.5,
        'n_estimators': 10,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'max_features': 0.5,
        'criterion': 'gini',

    }

    
    gullibility_threshold = st.slider("Gullibility Prediction Threshold", 0.0, 1.0, st.session_state.settings.get('gullibility_threshold', default_settings['gullibility_threshold']), 0.01)
    age_range = st.slider("Age Range", 18, 60, st.session_state.settings.get('age_range', default_settings['age_range']), 1)
    extraversion_threshold = st.slider("Extraversion Threshold", 0.0, 1.0, st.session_state.settings.get('extraversion_threshold', default_settings['extraversion_threshold']), 0.01)

    # RandomForestClassifier hyperparameters
    n_estimators = st.slider("Number of Trees (n_estimators)", 1, 100, st.session_state.settings.get('n_estimators', default_settings['n_estimators']), 1)
    max_depth = st.slider("Maximum Tree Depth (max_depth)", 1, 20, st.session_state.settings.get('max_depth', default_settings['max_depth']), 1)
    min_samples_split = st.slider("Minimum Samples Split (min_samples_split)", 2, 20, st.session_state.settings.get('min_samples_split', default_settings['min_samples_split']), 1)
    min_samples_leaf = st.slider("Minimum Samples Leaf (min_samples_leaf)", 1, 20, st.session_state.settings.get('min_samples_leaf', default_settings['min_samples_leaf']), 1)
    random_state = st.slider("Random Seed (random_state)", 1, 100, st.session_state.settings.get('random_state', default_settings['random_state']), 1)
    max_features = st.slider("Max Features (max_features)", 0.1, 1.0, st.session_state.settings.get('max_features', default_settings['max_features']), 0.1)
    criterion = st.selectbox("Criterion", ["gini", "entropy"], index=0 if st.session_state.settings.get('criterion', default_settings['criterion']) == 'gini' else 1)

    # Store settings in a dictionary
    settings_dict = {
        'gullibility_threshold': gullibility_threshold,
        'age_range': age_range,
        'extraversion_threshold': extraversion_threshold,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': random_state,
        'max_features': max_features,
        'criterion': criterion,

    }

    # Reset button
    if st.button("Reset Settings"):
        st.session_state.settings = default_settings
        settings_dict = default_settings
        st.success("Settings reset successfully.")


    st.session_state.settings.update(settings_dict)


# Feedback form page
def feedback_page():
    st.title("User Feedback")
    st.header(":mailbox: Get In touch with the GCM team for assistance")

    contact_form = """
    <form action="https://formsubmit.co/mufaronyagu@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="message" placeholder="Your message here"></textarea>
         <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)



# Main dashboard function
def main_dashboard():
    st.sidebar.title("Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    evaluate_model = st.sidebar.button("Evaluate Model")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        visualize_data(df)
        classifier = train_model(df)

        if evaluate_model:
            evaluate_classifier(classifier, df)

        # Using the settings for prediction
        predict_gullible(classifier, df, st.session_state.settings)

        # Link to feedback form page
        if st.sidebar.button("User Feedback"):
            st.experimental_rerun()
    else:
        st.sidebar.write("Please upload a CSV file.")

# Function to handle logout
def logout():
    st.session_state.login_status = False
    st.success("Logged out successfully.")

# Main function
def main():
    st.title("Gullibility Classification Model")
    st.write("By Mufaro Andre Nyagura v1.0")
    st.markdown('<marquee style="color:red;font-size:20px;">The Human Security Firewall</marquee>', unsafe_allow_html=True)

    # Initialising session state
    if "login_status" not in st.session_state:
        st.session_state.login_status = False

    # Initialising settings in session state
    if "settings" not in st.session_state:
        st.session_state.settings = {}

    # Checking login status
    if not st.session_state.login_status:
        login()
    else:
        # Creating sidebar for file upload, model evaluation, and logout
        st.sidebar.title("Dashboard")
        pages = ["Home", "Settings", "User Feedback"]
        selected_page = st.sidebar.selectbox("Select Page", pages)

        if selected_page == "Home":
            st.sidebar.write("Welcome to the Dashboard")
            main_dashboard()
        elif selected_page == "Settings":
            settings()
        elif selected_page == "User Feedback":
            feedback_page()

        # Logout button
        if st.sidebar.button("Logout"):
            logout()

if __name__ == '__main__':
    main()
