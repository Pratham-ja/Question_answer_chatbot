import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
import os
import pickle

# Sidebar
with st.sidebar:
    st.title("Chat with your PDF")
    st.markdown("""
    ### About
    This app uses **open-source models**  
    and runs **fully offline**.
    """)
    st.write("Made by Pratham Chaudhary")

def main():
    st.header("💬 Chat with your PDF (Open Source)")
    pdf = st.file_uploader("📄 Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Create or load FAISS vector store
        store_name = pdf.name[:-4]
        pkl_path = f"{store_name}_faiss.pkl"

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                vectorstore = pickle.load(f)
            st.info("✅ Loaded existing FAISS embeddings.")
        else:
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(pkl_path, "wb") as f:
                pickle.dump(vectorstore, f)
            st.success("✅ Created new FAISS embeddings.")

        # Query section
        query = st.text_input("🔍 Ask something about your PDF:")

        if query:
            docs = vectorstore.similarity_search(query=query, k=2)

            # Use a small local HuggingFace model for Q&A
            hf_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",  # Small, efficient, CPU-friendly
                max_new_tokens=256,
                temperature=0.3
            )
            llm = HuggingFacePipeline(pipeline=hf_pipeline)

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            st.write("### 🧠 Answer:")
            st.write(response)

if __name__ == "__main__":
    main()
