import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Quality Checker", layout="wide")
st.title("📊 RAG Quality Checker")

# ---------------- Session State ----------------
if "pipeline_answers" not in st.session_state:
    st.session_state.pipeline_answers = None

if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None


# ---------------- Upload PDF ----------------
st.header("1️⃣ Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file:
    if st.button("Upload & Index"):
        with st.spinner("Uploading and indexing PDF..."):
            res = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": uploaded_file},
            )

        if res.status_code == 200:
            st.success("PDF indexed successfully")
        else:
            st.error(res.text)


# ---------------- Ask Question ----------------
st.header("2️⃣ Ask Question")

question = st.text_input("Enter your question")

if st.button("Get Pipeline Answers"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Running RAG pipelines..."):
            res = requests.post(
                f"{BACKEND_URL}/ask",
                data={"question": question, "k": 4},
            )

        if res.status_code != 200:
            st.error(res.text)
        else:
            st.session_state.pipeline_answers = res.json()["pipelines"]
            st.session_state.evaluation_result = None
            st.success("Pipeline answers generated")


# ---------------- Show Pipeline Answers ----------------
if st.session_state.pipeline_answers:
    st.subheader("📌 Pipeline Answers")

    for name, info in st.session_state.pipeline_answers.items():
        with st.expander(name):
            st.markdown("**Answer:**")
            st.write(info["answer"])


# ---------------- Compare Pipelines ----------------
if st.session_state.pipeline_answers:
    st.header("3️⃣ Compare Pipelines")

    if st.button("Compare Pipelines"):
        with st.spinner("Evaluating pipeline quality..."):
            res = requests.post(
                f"{BACKEND_URL}/evaluate",
                data={"question": question, "k": 4},
            )

        if res.status_code != 200:
            st.error(res.text)
        else:
            st.session_state.evaluation_result = res.json()["evaluation"]


# ---------------- Show Evaluation ----------------
if st.session_state.evaluation_result:
    eval_data = st.session_state.evaluation_result

    st.subheader("🏆 Evaluation Result")

    st.markdown(
        f"""
        **Best Pipeline:** 🥇 `{eval_data['best_pipeline']}`  
        **Reason:** {eval_data['reason']}
        """
    )

    st.subheader("📊 Pipeline Scores")

    for name, score in eval_data["scores"].items():
        if name == eval_data["best_pipeline"]:
            st.success(f"{name} → ⭐ Score: {score}/10")
        else:
            st.info(f"{name} → Score: {score}/10")
