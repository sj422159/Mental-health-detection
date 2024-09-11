
# Mental Health Detector App

![Mental Health Detector](https://img.shields.io/badge/Mental_Health-Detection-brightgreen.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg) ![Machine Learning](https://img.shields.io/badge/ML-Logistic_Regression-blue.svg)

### A web application for detecting mental health conditions using text analysis and machine learning.

## 🧠 Overview

The **Mental Health Detector** app analyzes text input to predict potential mental health conditions such as **Schizophrenia**, **Borderline Personality Disorder (BPD)**, and general **Mental Health Issues**. This tool utilizes a machine learning model built with logistic regression to classify the mental health status based on user-submitted text.

The app is designed to help monitor mental health by providing insights based on language usage patterns. This is **not a diagnostic tool**, but it may aid in recognizing patterns that could prompt further professional consultation.

## 🚀 Features

- **Text-based Predictions**: Enter text and get predictions related to mental health conditions.
- **Confidence Scores**: See the confidence level of each prediction.
- **Interactive Data Visualization**: Visualize the prediction probabilities using an interactive bar chart.
- **Easy to Use**: Simple interface with a sidebar menu for easy navigation.
- **Fast and Lightweight**: Powered by Streamlit for quick, browser-based access.

## 📸 Screenshots

![App Screenshot](screenshot1.png)
*Example of mental health text analysis with prediction and confidence scores.*

## 🔧 Installation

To run the **Mental Health Detector** app locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/sj422159/mental-health-detector.git
cd mental-health-detector
```

### 2. Install dependencies

Make sure you have Python 3.x installed. Then install the required Python packages using:

```bash
pip install -r requirements.txt
```

### 3. Run the app

Once the dependencies are installed, run the Streamlit app with:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🧩 Dependencies

The app requires the following Python libraries:

- **Streamlit**: For building the web interface.
- **Scikit-learn**: For loading and using the logistic regression model.
- **Altair**: For data visualization (prediction probabilities).
- **Joblib**: For loading the saved machine learning model.

You can install all dependencies by running the command:

```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Launch the app.
2. Select **Home** from the sidebar menu.
3. Type your text in the input box (e.g., journal entries, thoughts, or feelings).
4. Click the **Submit** button.
5. View the predicted mental health condition and the confidence score.
6. Explore the prediction probabilities using the bar chart visualization.

## 📝 Example Texts for Testing

- **Schizophrenia**: "I hear voices sometimes, and I feel like someone is controlling my thoughts."
- **Mental Health**: "I've been feeling overwhelmed and anxious lately, and it's hard to concentrate on anything."
- **Borderline Personality Disorder (BPD)**: "My emotions change so quickly, and I often feel like people are going to abandon me."

## 🎯 Model

The model used in this app is a **Logistic Regression** classifier trained on mental health-related text data. The model predicts based on linguistic patterns associated with various mental health conditions.

## 👨‍💻 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/sj422159/mental-health-detector/issues).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🛡️ Disclaimer

This app is **not intended for diagnostic purposes**. It is a tool for educational and informational use only. Always consult a licensed mental health professional for any mental health concerns.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

#   M e n t a l - h e a l t h - d e t e c t i o n  
 