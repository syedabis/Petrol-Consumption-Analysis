import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="Petrol Consumption Analysis", layout="wide")

# Title and Description
st.title("Petrol Consumption Analysis & Prediction")
st.markdown("""
This app visualizes the Petrol Consumption dataset and allows you to predict petrol consumption based on various factors.
""")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
        return df
    except FileNotFoundError:
        st.error("data.csv not found. Please ensure the file is in the same directory.")
        return None

df = load_data()

if df is not None:
    # Sidebar for content navigation
    st.sidebar.header("Navigation")
    options = st.sidebar.radio("Go to:", ["Data Overview", "Visualizations", "Prediction"])

    if options == "Data Overview":
        st.header("Data Overview")
        if st.checkbox("Show Raw Data"):
            st.dataframe(df)
        
        st.subheader("Statistical Summary")
        st.write(df.describe())

    elif options == "Visualizations":
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Distribution of Petrol Consumption")
            fig, ax = plt.subplots()
            sns.histplot(df["Petrol_Consumption"], kde=True, ax=ax)
            st.pyplot(fig)
            
        st.subheader("Feature Relationships")
        feature = st.selectbox("Select Feature to compare with Petrol Consumption", 
                               [col for col in df.columns if col != "Petrol_Consumption"])
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature, y="Petrol_Consumption", ax=ax)
        ax.set_title(f"Petrol Consumption vs {feature}")
        st.pyplot(fig)

    elif options == "Prediction":
        st.header("Predict Petrol Consumption")
        st.markdown("Train a Linear Regression model to predict petrol consumption.")

        # Prepare Data
        X = df.drop("Petrol_Consumption", axis=1)
        y = df["Petrol_Consumption"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Model Metrics
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"**Model Performance (R² Score):** {r2:.2f}")

        st.subheader("Make a Prediction")
        
        # User Inputs
        input_data = {}
        for col in X.columns:
            val = st.number_input(f"Enter {col}", value=float(df[col].mean()))
            input_data[col] = val
            
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Petrol Consumption: **{prediction:.2f}**")
