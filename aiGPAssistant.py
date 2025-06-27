import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import io
from datetime import datetime

# Initialize the Mistral client
api_key = ""
client_api = MistralClient(api_key=api_key)

def get_text_embedding(input):
    embeddings_batch_response = client_api.embeddings(
        model="mistral-embed",
        input=input
    )
    return embeddings_batch_response.data[0].embedding

def get_embeddings_by_chunks(data, chunk_size):
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client_api.embeddings(model="mistral-embed", input=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data]

def run_mistral(user_message, model="mistral-medium"):
    client = MistralClient(api_key=api_key)
    messages = [
        ChatMessage(role="user", content=user_message)
    ]
    chat_response = client.chat(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content

def get_medical_description(diagnosis):
    user_message = f"Provide a short medical description for {diagnosis} keep it clear, crisp and to the point also describing its most common cause or set of causes. Add top 5 symptoms and always advice the to seek medical advice as soon as possible."
    return run_mistral(user_message)

@st.cache_data
def load_data(url=None, file=None):
    if url:
        df = pd.read_csv(url, index_col=0)
    elif file:
        df = pd.read_csv(file, index_col=0)
    return df

def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'clf' not in st.session_state:
        st.session_state.clf = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = LabelEncoder()
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    if 'diagnosis' not in st.session_state:
        st.session_state.diagnosis = ""
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0.0
    if 'tsne_fig' not in st.session_state:
        st.session_state.tsne_fig = None
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'confidence_thresholds' not in st.session_state:
        st.session_state.confidence_thresholds = {
            'green': [75, 100],
            'amber': [55, 74],
            'red': [0, 54]
        }
    if 'medical_description' not in st.session_state:
        st.session_state.medical_description = ""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "Step 1: Data Ingestion"

initialize_session_state()

st.title("GP's Assistant Diagnostician")

# Sidebar navigation tiles
with st.sidebar:
    st.write("### Navigation Menu")
    if st.button("Step 1: Data Ingestion"):
        st.session_state.current_step = "Step 1: Data Ingestion"
    if st.button("Step 2: Process Data"):
        st.session_state.current_step = "Step 2: Process Data"
    if st.button("Step 3: Show Clusters"):
        st.session_state.current_step = "Step 3: Show Clusters"
    if st.button("Step 4: Train Agent"):
        st.session_state.current_step = "Step 4: Train Agent"
    if st.button("Step 5: Enter Symptoms and Diagnose"):
        st.session_state.current_step = "Step 5: Enter Symptoms and Diagnose"
    if st.button("Search History"):
        st.session_state.current_step = "Search History"
    if st.button("Customize Confidence Thresholds"):
        st.session_state.current_step = "Customize Confidence Thresholds"
    
    # Clear all outputs
    if st.button("Clear All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

def add_info_icon(step_name, info_text):
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <h3 style="margin: 0;">{step_name}</h3>
        <span style="margin-left: 5px; cursor: pointer;" title="{info_text}">
            ℹ️
        </span>
    </div>
    """, unsafe_allow_html=True)

# Functions to display each step
def step_1():
    add_info_icon("Step 1: Data Ingestion", "Select the data source (URL or File Upload) and load the CSV file.")
    data_source = st.radio("Select data source:", ("URL", "File Upload"), key="data_source")
    if data_source == "URL":
        data_url = st.text_input("Enter the URL of the CSV file:")
        if st.button("Show Data"):
            st.session_state.df = load_data(url=data_url)
    elif data_source == "File Upload":
        data_file = st.file_uploader("Upload CSV file", type="csv")
        if st.button("Show Data"):
            st.session_state.df = load_data(file=data_file)

    if st.session_state.df is not None:
        with st.expander("Show Data", expanded=True):
            st.dataframe(st.session_state.df)

def step_2():
    add_info_icon("Step 2: Process Data", "Process the loaded data to generate embeddings using the Mistral API.")
    if st.button("Process"):
        if st.session_state.df is not None:
            with st.spinner("Processing embeddings..."):
                st.session_state.df["embeddings"] = get_embeddings_by_chunks(st.session_state.df["text"].tolist(), 50)
                st.session_state.df["encoded_labels"] = st.session_state.label_encoder.fit_transform(st.session_state.df["label"])
                st.session_state.processed = True
            st.success("Embeddings have been processed.")

    if st.session_state.processed:
        with st.expander("Processed Data", expanded=True):
            st.dataframe(st.session_state.df)
            processed_data = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed Data",
                data=processed_data,
                file_name='processed_data.csv',
                mime='text/csv',
            )

def step_3():
    add_info_icon("Step 3: Show Clusters", "Run t-SNE to reduce dimensions and visualize the data clusters.")
    if st.button("Show Clusters"):
        if st.session_state.processed:
            with st.spinner("Running Cluster Analysis..."):
                tsne = TSNE(n_components=3, random_state=0).fit_transform(np.array(st.session_state.df['embeddings'].to_list()))
                st.session_state.df['tsne-3d-one'] = tsne[:, 0]
                st.session_state.df['tsne-3d-two'] = tsne[:, 1]
                st.session_state.df['tsne-3d-three'] = tsne[:, 2]

                st.session_state.tsne_fig = px.scatter_3d(
                    st.session_state.df, x='tsne-3d-one', y='tsne-3d-two', z='tsne-3d-three',
                    color='label',
                    hover_data=['text']
                )
                st.plotly_chart(st.session_state.tsne_fig)

    if st.session_state.tsne_fig is not None:
        with st.expander("Cluster Visualization", expanded=True):
            st.plotly_chart(st.session_state.tsne_fig)
            fig_io = io.BytesIO()
            st.session_state.tsne_fig.write_image(fig_io, format='png')
            st.download_button(
                label="Download Cluster Visualization",
                data=fig_io,
                file_name='cluster_visualization.png',
                mime='image/png',
            )

def step_4():
    add_info_icon("Step 4: Train Agent", "Train a Logistic Regression model using the processed embeddings.")
    if st.button("Train Agent"):
        if st.session_state.processed:
            with st.spinner("Training Agent..."):
                train_x, test_x, train_y, test_y = train_test_split(
                    st.session_state.df["embeddings"], st.session_state.df["encoded_labels"], test_size=0.2
                )
                scaler = StandardScaler()
                train_x = scaler.fit_transform(train_x.tolist())
                test_x = scaler.transform(test_x.tolist())

                # Store the scaler in session state
                st.session_state.scaler = scaler

                clf = LogisticRegression(random_state=0, max_iter=500)
                clf.fit(train_x, train_y)

                st.session_state.clf = clf

                # Evaluate the model
                accuracy = clf.score(test_x, test_y)
                precision = np.mean(clf.predict(test_x) == test_y)

                st.session_state.model_metrics = {'Accuracy': accuracy, 'Precision': precision}

                st.write(f"Model Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")

    if st.session_state.model_metrics:
        st.write("### Model Performance Metrics")
        cols_metrics = st.columns(len(st.session_state.model_metrics))
        for col, (metric, value) in zip(cols_metrics, st.session_state.model_metrics.items()):
            col.metric(metric, f"{value:.2f}")

def step_5():
    add_info_icon("Step 5: Enter Symptoms and Diagnose", "Enter symptoms to get a diagnosis from the trained model.")
    symptoms_input = st.text_input("Enter your symptoms:", st.session_state.user_input)
    if st.button("Diagnose") and symptoms_input and st.session_state.clf is not None:
        embedding = get_text_embedding([symptoms_input])

        # Apply the scaler from the training step
        embedding = st.session_state.scaler.transform([embedding])

        prediction_encoded = st.session_state.clf.predict(embedding)[0]
        prediction = st.session_state.label_encoder.inverse_transform([prediction_encoded])[0]

        confidence = st.session_state.clf.predict_proba(embedding)[0]
        confidence_score = np.max(confidence)

        st.session_state.diagnosis = prediction
        st.session_state.confidence = confidence_score

        # Save to search history
        st.session_state.search_history.append({
            "input_text": symptoms_input,
            "prediction": prediction,
            "confidence": confidence_score,
            "DT": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Get medical description
        with st.spinner("Generating medical description..."):
            st.session_state.medical_description = get_medical_description(prediction)

        # Reset user input
        st.session_state.user_input = ""

    # Display diagnosis, confidence, and medical description
    if st.session_state.diagnosis:
        st.write("### Diagnosis")
        st.markdown(f"<div style='display: flex; align-items: center;'><div style='background-color: #f0f0f5; border-radius: 20px; padding: 10px; margin-right: 10px;'>{st.session_state.diagnosis}</div><div>", unsafe_allow_html=True)
        st.metric("Confidence", f"{st.session_state.confidence:.2f}")
        st.write("### Medical Description")
        st.write(st.session_state.medical_description)

def search_history():
    st.subheader("Search History")
    with st.expander("Show Search History", expanded=True):
        if st.session_state.search_history:
            for entry in st.session_state.search_history:
                confidence_color = get_confidence_color(entry['confidence'] * 100)
                st.markdown(
                    f"""
                    <div style="background-color: {confidence_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>Input Text:</strong> {entry['input_text']}<br>
                        <strong>Prediction:</strong> {entry['prediction']}<br>
                        <strong>Confidence:</strong> {entry['confidence']:.2f}<br>
                        <strong>Timestamp:</strong> {entry['DT']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Save search history as CSV
            search_history_df = pd.DataFrame(st.session_state.search_history)
            search_history_csv = search_history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Search History",
                data=search_history_csv,
                file_name='search_history.csv',
                mime='text/csv',
            )

            if st.button("Analyze"):
                analyze_search_history()
        else:
            st.write("No search history available.")

# Function to get color based on confidence
def get_confidence_color(confidence):
    thresholds = st.session_state.confidence_thresholds
    if thresholds['green'][0] <= confidence <= thresholds['green'][1]:
        return 'green'
    elif thresholds['amber'][0] <= confidence <= thresholds['amber'][1]:
        return 'amber'
    else:
        return 'red'

# Analyze function to generate interactive number cards
def analyze_search_history():
    if not st.session_state.search_history:
        st.write("No search history available.")
        return

    search_history_df = pd.DataFrame(st.session_state.search_history)
    unique_conditions = search_history_df['prediction'].unique()

    analysis = []
    for condition in unique_conditions:
        condition_data = search_history_df[search_history_df['prediction'] == condition]
        green_count = sum(condition_data['confidence'] * 100 >= st.session_state.confidence_thresholds['green'][0])
        amber_count = sum((condition_data['confidence'] * 100 >= st.session_state.confidence_thresholds['amber'][0]) & (condition_data['confidence'] * 100 < st.session_state.confidence_thresholds['green'][0]))
        red_count = sum(condition_data['confidence'] * 100 < st.session_state.confidence_thresholds['amber'][0])
        total = len(condition_data)
        analysis.append({
            "condition": condition,
            "green": green_count / total * 100 if total > 0 else 0,
            "amber": amber_count / total * 100 if total > 0 else 0,
            "red": red_count / total * 100 if total > 0 else 0
        })

    for entry in analysis:
        st.write(f"### {entry['condition']}")
        st.metric("Green", f"{entry['green']:.2f}%")
        st.metric("Amber", f"{entry['amber']:.2f}%")
        st.metric("Red", f"{entry['red']:.2f}%")

def customize_confidence_thresholds():
    st.sidebar.header("Confidence Meter")
    with st.sidebar.expander("Customize Confidence Thresholds", expanded=True):
        thresholds = st.session_state.confidence_thresholds

        green_range = st.slider("Green Confidence Range (%)", 0, 100, (thresholds['green'][0], thresholds['green'][1]))
        amber_range = st.slider("Amber Confidence Range (%)", 0, 100, (thresholds['amber'][0], thresholds['amber'][1]))
        red_range = st.slider("Red Confidence Range (%)", 0, 100, (thresholds['red'][0], thresholds['red'][1]))

        if green_range != tuple(thresholds['green']):
            st.session_state.confidence_thresholds['green'] = list(green_range)
        if amber_range != tuple(thresholds['amber']):
            st.session_state.confidence_thresholds['amber'] = list(amber_range)
        if red_range != tuple(thresholds['red']):
            st.session_state.confidence_thresholds['red'] = list(red_range)

    with st.sidebar.expander("Model Output History", expanded=False):
        if st.session_state.search_history:
            for entry in st.session_state.search_history:
                confidence_color = get_confidence_color(entry['confidence'] * 100)
                st.markdown(
                    f"""
                    <div style="background-color: {confidence_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>Input Text:</strong> {entry['input_text']}<br>
                        <strong>Prediction:</strong> {entry['prediction']}<br>
                        <strong>Confidence:</strong> {entry['confidence']:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.write("No confidence data available.")

# Display the selected step
current_step = st.session_state.current_step
if current_step == "Step 1: Data Ingestion":
    step_1()
elif current_step == "Step 2: Process Data":
    step_2()
elif current_step == "Step 3: Show Clusters":
    step_3()
elif current_step == "Step 4: Train Agent":
    step_4()
elif current_step == "Step 5: Enter Symptoms and Diagnose":
    step_5()
elif current_step == "Search History":
    search_history()
elif current_step == "Customize Confidence Thresholds":
    customize_confidence_thresholds()
