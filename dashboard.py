import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Quality Checker", layout="wide")
st.title("RAG quality checker")

st.header("upload a PDF")
uploaded_file = st.file_uploader("choose a pdf", type=["pdf"])

if uploaded_file:
    if st.button("Upload & Index"):
        with st.spinner("uploading and indexing"):
            files = {"file":uploaded_file}
            res = requests.post(f"{BACKEND_URL}/upload", files=files)

            if res.status_code == 200:
                st.success("PDF indexed sucessfully")
            else:
                st.error(res.text)


st.header("Ask Question")
question = st.text_input("Enter your question")

if st.button("evaluate pipelines"):
    if not question:
        st.warning("please enter a question")

    else:
        with st.spinner("running rag + evaluator"):
            data = {
                "question" : question,
                "k":4
            }
            res = requests.post(f"{BACKEND_URL}/evaluate", data=data)

        if res.status_code != 200:
            st.error(res.text)
        else:
            result = res.json()
            st.subheader("evaluation result")

            st.subheader("pipeline results")

            for name, info in result["pipelines"].items():
                with st.expander(name):
                    st.markdown("**Answer:**")
                    st.write(info["answer"])




