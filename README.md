# GP's Assistant Diagnostician

**GP's Assistant Diagnostician** is a next-generation, interactive Streamlit application that combines the power of Large Language Models (LLMs) with traditional machine learning to deliver fast, explainable, and actionable medical predictions from minimal, human-readable data.


## Walkthrough



https://github.com/user-attachments/assets/de7aef80-fd82-4c1d-8c87-3718edcdffb1



---

## 🚀 What Makes This App Unique?

- **Minimal Data, Maximum Insight:**  
  Works with small datasets—just disease labels and free-text symptoms—making it ideal for rapid prototyping and real-world scenarios where data is scarce.

- **Hybrid Intelligence:**  
  Leverages state-of-the-art LLM embeddings (via Mistral API) to transform natural language symptoms into rich vector representations, then applies classic ML (Logistic Regression) for robust, interpretable predictions.

- **End-to-End Workflow:**  
  - **Data Ingestion:** Load CSVs from URL or file upload.
  - **Embedding Generation:** Convert symptoms to embeddings using LLMs.
  - **Clustering & Visualization:** Explore your data with interactive 3D t-SNE plots.
  - **Model Training:** Train and evaluate a classifier in one click.
  - **Diagnosis:** Enter symptoms in plain English and receive a prediction, confidence score, and a concise, LLM-generated medical description.
  - **Search History & Download:** Every diagnosis is logged for review, analysis, and export—perfect for reinforcement learning or audit trails.

- **Customizable Confidence Meter:**  
  Adjust thresholds for green/amber/red confidence bands to match your risk tolerance or clinical needs.

- **Modern, Intuitive UI:**  
  Step-by-step navigation, info tooltips, and one-click downloads for processed data, cluster visualizations, and search history.

---

## 🏁 Quick Start

1. **Clone the repository and install requirements:**
    ```sh
    git clone https://github.com/yourusername/gp-assistant-diagnostician.git
    cd gp-assistant-diagnostician
    pip install -r requirements.txt
    ```

2. **Set your Mistral API key:**
    - Edit `aiGPAssistant.py` and replace the placeholder with your Mistral API key.

3. **Run the app:**
    ```sh
    streamlit run aiGPAssistant.py
    ```

---

## 🩺 Typical Workflow

1. **Step 1: Data Ingestion**  
   Upload or link to a CSV with two columns: `label` (disease) and `text` (symptoms).

2. **Step 2: Process Data**  
   Generate embeddings for each symptom entry using the Mistral LLM API.

3. **Step 3: Show Clusters**  
   Visualize your dataset in 3D using t-SNE and Plotly.

4. **Step 4: Train Agent**  
   Train a Logistic Regression model on the embeddings and see performance metrics.

5. **Step 5: Enter Symptoms and Diagnose**  
   Type symptoms in plain English. Get a prediction, confidence score, and a crisp medical summary (LLM-generated).

6. **Search History & Analysis**  
   Review, analyze, and download all past diagnoses for further learning or compliance.

---

## 📦 Features

- **Drag-and-drop or URL-based data ingestion**
- **LLM-powered embeddings for natural language**
- **Interactive 3D cluster visualization**
- **One-click model training and evaluation**
- **Natural language diagnosis with confidence scoring**
- **LLM-generated medical explanations**
- **Downloadable processed data, cluster images, and search history**
- **Customizable confidence thresholds**
- **Session state for seamless navigation**

---

## 🔒 Privacy & Security

- All processing is local except for secure calls to the Mistral API for embeddings and LLM responses.
- No patient-identifiable data is required or stored.

---

## 📈 For Developers & Researchers

- **Reinforcement Learning Ready:**  
  Download search history as CSV for further model tuning or RL pipelines.
- **Easily Extendable:**  
  Swap in your own LLM or ML model with minimal code changes.

---

## 📝 Example Data Format

| label         | text                                 |
|---------------|--------------------------------------|
| Influenza     | fever, cough, sore throat, headache  |
| Migraine      | severe headache, nausea, light sensitivity |

---

## 📞 Contact

For questions, suggestions, or contributions, please open an issue or contact the maintainer.



Empowering clinicians and researchers to do more with less—combining the best of LLMs and classic ML for smarter, faster, and more transparent
