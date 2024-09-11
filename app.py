import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Load model
pipe_lr = joblib.load(open(r'./logisticmentalhealthmodel.pkl', 'rb'))

# Function to predict mental health
def predict_mhealth(docs):
    results = pipe_lr.predict([docs])
    return results[0]

# Function to get prediction probabilities
def get_predictions_proba(docs):
    results = pipe_lr.predict_proba([docs])
    return results

# Main function for the Streamlit app
def main():
    st.title("Mental Health Detector")
    
    # Sidebar menu
    menu = ['Home', 'Monitor', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home - Mental Health Detection')

        # Input form for text data
        with st.form(key='Mental_clf'):
            raw_text = st.text_area('Type text here')
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            # Columns for showing the results
            col1, col2 = st.columns(2)

            # Get prediction and probability
            prediction = predict_mhealth(raw_text)
            probability = get_predictions_proba(raw_text)

            # Column 1: Display original text and prediction
            with col1:
                st.success('Original Text')
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)
                st.write(f'Confidence: {np.max(probability):.2f}')

            # Column 2: Display prediction probability chart
            with col2:
                st.success('Prediction Probability')

                # Convert probabilities into a dataframe
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['Mental_Health', 'Probability']

                # Create a bar chart using Altair
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='Mental_Health',
                    y='Probability',
                    color='Mental_Health'
                )
                st.altair_chart(fig, use_container_width=True)

    elif choice == 'Monitor':
        st.subheader('Monitor App')
  

    else:
        st.subheader('About')
        st.write('This is a simple mental health detection app using text data.')


if __name__ == '__main__':
    main()
