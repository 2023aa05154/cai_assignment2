# ## Conversational AI Assignment 2
# ### Group : 70
# ### Members:

# - ATUL SHARMA
# - MONISH. K.B
# - SHIVAM RAVI
# - VENKATRAMAN SOUMYA P G VENKATRAMAN
# - VOBBANI VENKATESWARLU

# ### Problem Statement:
# - Implement Basic RAG
# - Implement Advanced RAG using `Memory-Augmented Retrieval`


# RAG Implementations START #############################################################################

# Imports
import streamlit as st
import numpy as np
import os
import tempfile
import torch
from torch import cuda

# Fix for torch issue with streamlit package (ref: https://github.com/VikParuchuri/marker/issues/442)
torch.classes.__path__ = [
    os.path.join(torch.__path__[0], torch.classes.__file__)]


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


os.environ['NLTK_DATA'] = "./nltk_data"

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import datetime
from Group_70_llm_utils import Utils


# Configurations
# Set page config
st.set_page_config(page_title="Financial Document Q&A", layout="wide")

# Define constants
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
# Small, efficient model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Small LLM for generation
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# RAG Prompt
rag_prompt = """
Based on the following financial document excerpts, please answer the question.

Question: {query}

Context:
{context}


If above context is not relevant to answer the query, simply reply "This information is not available in the context"

Answer:"""


# Memory for previously seen documents and queries
if 'document_memory' not in st.session_state:
    st.session_state.document_memory = []
if 'query_memory' not in st.session_state:
    st.session_state.query_memory = []


def log(text):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} : {__name__} : {text}")


class RagTypes:
    basic: str = "Basic RAG"
    advanced: str = "Advanced RAG"


# Document Processor (Splits and Embeds the documents)
class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.embedding_model = Utils.embedding_model

    def process_pdf(self, pdf_file):
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name

        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        os.unlink(pdf_path)  # Remove temp file

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Get text content from chunks
        texts = [doc.page_content for doc in chunks]

        # Add to memory
        st.session_state.document_memory.extend(texts)

        # Create embeddings
        embeddings = self.embedding_model.get_embeddings(texts)

        return texts, embeddings


# Instantiate global Response Generator and Embedding Model so that
# these memory heavy objects don't re-initialize on each strealit re-run
if not st.session_state.get("rag_initialized"):
    st.session_state.resp_gen = Utils.response_generator
    st.session_state.emb_mdl = Utils.embedding_model
    st.session_state.rag_initialized = True


# Basic RAG implementation (Data->Chunk->Embed->Retrieve->Answer)
class BasicRAG:
    def __init__(self):
        self.embedding_model = st.session_state.emb_mdl
        self.llm = st.session_state.resp_gen

    def retrieve(self, query, texts, embeddings, top_k):
        query_embedding = self.embedding_model.get_embeddings([query])

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Get top k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_chunks = [texts[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        return top_chunks, top_scores

    def answer(self, query, texts, embeddings, top_k):
        log(f"Generating answer using top k: {top_k}")
        relevant_chunks, scores = self.retrieve(
            query, texts, embeddings, top_k)

        # Concatenate chunks into context
        context = "\n\n".join(relevant_chunks)

        # Generate answer
        answer = self.llm.generate_response(query, context)

        # Calculate confidence score (average similarity of top chunks)
        confidence = sum(scores) / len(scores) if scores else 0

        return answer, confidence, relevant_chunks, scores


# Basic RAG implementation (Uses Memory Augmented RAG + hybrid search)
class AdvancedRAG:
    def __init__(self):
        self.embedding_model = st.session_state.emb_mdl
        self.llm = st.session_state.resp_gen

    def setup_bm25(self, texts):
        # Tokenize texts for BM25
        tokenized_corpus = []
        for text in texts:
            tokenized_text = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            tokenized_text = [word for word in tokenized_text if word.isalnum() and word not in stop_words]
            tokenized_corpus.append(tokenized_text)

        return BM25Okapi(tokenized_corpus), tokenized_corpus

    def retrieve(self, query, texts, embeddings, top_k):
        # Semantic search with embeddings
        query_embedding = self.embedding_model.get_embeddings([query])
        semantic_similarities = cosine_similarity(query_embedding, embeddings)[0]

        # BM25 keyword search
        bm25, tokenized_corpus = self.setup_bm25(texts)
        tokenized_query = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        tokenized_query = [word for word in tokenized_query if word.isalnum() and word not in stop_words]
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize scores
        if max(semantic_similarities) > 0:
            semantic_similarities = semantic_similarities / max(semantic_similarities)
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)

        # Hybrid scoring (adjust weights as needed)
        hybrid_scores = 0.7 * semantic_similarities + 0.3 * bm25_scores

        # Re-ranking: add bonus for documents containing financial terms
        financial_terms = ['revenue', 'profit', 'earnings', 'balance', 'asset', 'liability', 
                          'income', 'expense', 'financial', 'quarter', 'fiscal', 'dividend']

        for i, text in enumerate(texts):
            text_lower = text.lower()
            term_matches = sum(1 for term in financial_terms if term in text_lower)
            # Add a small bonus for each financial term
            hybrid_scores[i] += 0.02 * term_matches

        # Memory-augmented retrieval: boost scores for chunks similar to past successful queries
        if st.session_state.query_memory:
            for past_query in st.session_state.query_memory:
                past_embedding = self.embedding_model.get_embeddings([past_query])
                past_similarities = cosine_similarity(past_embedding, embeddings)[0]
                # Add a small boost based on similarity to past queries
                hybrid_scores += 0.1 * past_similarities

        # Get top k chunks
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        top_chunks = [texts[i] for i in top_indices]
        top_scores = [hybrid_scores[i] for i in top_indices]

        return top_chunks, top_scores

    def calculate_confidence(self, query, answer, chunks, scores):
        # Base confidence from retrieval scores
        retrieval_confidence = sum(scores) / len(scores) if scores else 0

        # Check answer quality factors
        answer_length_factor = min(len(answer) / 100, 1.0)  # Penalize very short answers

        # Check if answer contains numbers (financial answers often do)
        contains_numbers = bool(re.search(r'\d', answer))
        number_bonus = 0.1 if contains_numbers else 0

        # Check if the answer mentions uncertainty
        uncertainty_terms = ['unclear', 'unknown', 'uncertain', 'not specified', 'cannot determine', 
                             'not mentioned', 'insufficient information']
        uncertainty_penalty = 0.2 if any(
            term in answer.lower() for term in uncertainty_terms) else 0

        # Final confidence calculation
        confidence = (
            0.6 * retrieval_confidence + 
            0.2 * answer_length_factor + 
            number_bonus - 
            uncertainty_penalty
        )

        # Clamp between 0 and 1
        return max(0, min(confidence, 1))

    def answer(self, query, texts, embeddings, top_k):
        log(f"Generating answer using top k: {top_k}")
        relevant_chunks, scores = self.retrieve(
            query, texts, embeddings, top_k)

        # Concatenate chunks into context with weighting by score
        weighted_chunks = []
        for chunk, score in zip(relevant_chunks, scores):
            weighted_chunks.append(f"[Relevance: {score:.2f}] {chunk}")

        context = "\n\n".join(weighted_chunks)

        # Generate answer
        answer = self.llm.generate_response(query, context)

        # Calculate confidence score
        confidence = self.calculate_confidence(
            query, answer, relevant_chunks, scores)

        # Add successful query to memory
        if confidence > 0.6:
            if query not in st.session_state.query_memory:
                st.session_state.query_memory.append(query)
                # Keep memory limited to recent queries
                if len(st.session_state.query_memory) > 10:
                    st.session_state.query_memory.pop(0)

        return answer, confidence, relevant_chunks, scores

# RAG Implementations END ###############################################################################


# Streamlit UI Implementations START ####################################################################


log("Initializing the application...")
st.title("Financial Document Q&A")


# Initialize Basic and Advanced Rag if not initialized
if 'basic_rag' not in st.session_state:
    st.session_state.basic_rag = BasicRAG()
if 'advanced_rag' not in st.session_state:
    st.session_state.advanced_rag = AdvancedRAG()
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'answer' not in st.session_state:
    st.session_state.answer = {}
if 'answer_k' not in st.session_state:
    st.session_state.answer_k = 2
if 'answer_rag_type' not in st.session_state:
    st.session_state.answer_rag_type = RagTypes.basic
if 'test_res_high' not in st.session_state:
    st.session_state.test_res_high = {}
if 'test_res_low' not in st.session_state:
    st.session_state.test_res_low = {}


# Sidebar for configuration
st.sidebar.title("Configuration")
rag_type = st.sidebar.radio(
    "RAG Type", [RagTypes.basic, RagTypes.advanced], index=0)
chunk_size = st.sidebar.slider("Chunk Size", 500, 1000, 700, 100)
top_k = st.sidebar.slider("Top K Chunks", 1, 5, 2, 1)

# Document upload section
st.header("Upload Financial Documents")
uploaded_files = st.file_uploader(
    "Upload PDF files", type=['pdf'], accept_multiple_files=True)

# Process uploaded documents
if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor(chunk_size=chunk_size)

if 'texts' not in st.session_state:
    st.session_state.texts = []
    st.session_state.embeddings = None
log('Application Initialized.')

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_texts = []
        all_embeddings = []

        for file in uploaded_files:
            log(f'Embedding file: {file.name}')
            texts, embeddings = st.session_state.processor.process_pdf(
                file)
            all_texts.extend(texts)

            if len(all_embeddings) == 0:
                all_embeddings = embeddings
            else:
                all_embeddings = np.vstack((all_embeddings, embeddings))

        st.session_state.texts = all_texts
        st.session_state.embeddings = all_embeddings

        st.success(f"Processed {len(all_texts)} text chunks from "
                    f"{len(uploaded_files)} documents")

# Q&A section
st.header("Ask Questions")
query = st.text_input("Enter your question about the financial documents")

if (
    query and
    (
        not (
            (query == st.session_state.question) and
            (top_k == st.session_state.answer_k) and
            (rag_type == st.session_state.answer_rag_type)
        )
    ) and
    st.session_state.texts and st.session_state.embeddings is not None
        ):
    with st.spinner("Generating answer..."):

        if rag_type == RagTypes.basic:
            log(f"Generating resonse using {RagTypes.basic} for query: {query}")
            answer, confidence, chunks, scores = st.session_state.basic_rag.answer(
                query, st.session_state.texts, st.session_state.embeddings, top_k
            )
            st.session_state.answer_rag_type = RagTypes.basic
        else:
            log(f"Generating resonse using {RagTypes.advanced} for query: {query}")
            answer, confidence, chunks, scores = st.session_state.advanced_rag.answer(
                query, st.session_state.texts, st.session_state.embeddings, top_k
            )
            st.session_state.answer_rag_type = RagTypes.advanced
        st.session_state.answer["answer"] = answer
        st.session_state.answer["confidence"] = confidence
        st.session_state.answer["chunks"] = chunks
        st.session_state.answer["scores"] = scores
        st.session_state.question = query
        st.session_state.answer_k = top_k


# Display results
ans = st.session_state.answer
if ans:
    answer = ans['answer']
    confidence = ans['confidence']
    chunks = ans['chunks']
    scores = ans['scores']

    st.subheader("Answer")
    st.markdown(answer)
    st.subheader(f"Confidence Score: {confidence:.2f}")
    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
    st.markdown(f"<div style='width:{int(confidence*100)}%;height:20px;background-color:{confidence_color};'></div>", 
                unsafe_allow_html=True)

    with st.expander("View Relevant Document Chunks"):
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            st.markdown(f"**Chunk {i+1}** (Relevance: {score:.2f})")
            st.text(chunk)

expanded = (st.session_state.test_res_low is not None) or (st.session_state.test_res_high is not None)

# Testing section
if st.session_state.texts and st.session_state.embeddings is not None:
    st.header("Run Test Cases")

with st.expander('Execute Test Cases',expanded=expanded):
    if st.session_state.texts and st.session_state.embeddings is not None:
        test_col1, test_col2 = st.columns(2)
        with test_col1:
            test_query = "What is the Integrated yield ramp revenue for Q3, 2024?"
            st.text_input("Test Query", value=test_query, key="high_conf_query")
            if st.button("Test High-Confidence Query"):
                log("Running Test case with High confidence query")
                with st.spinner("Testing..."):
                    if rag_type == RagTypes.basic:
                        answer, confidence, chunks, scores = st.session_state.basic_rag.answer(
                            test_query, st.session_state.texts, st.session_state.embeddings, top_k
                        )
                    else:
                        answer, confidence, chunks, scores = st.session_state.advanced_rag.answer(
                            test_query, st.session_state.texts, st.session_state.embeddings, top_k
                        )
                    st.session_state.test_res_high['answer'] = answer
                    st.session_state.test_res_high["confidence"] = confidence
                    st.session_state.test_res_high["chunks"] = chunks
                    st.session_state.test_res_high["scores"] = scores
            test_res_high = st.session_state.test_res_high
            if test_res_high:
                answer = test_res_high['answer']
                confidence = test_res_high['confidence']
                st.markdown(f"**Answer:** {answer}")
                st.markdown(f"### Confidence Score: {confidence:.2f}")
                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.markdown(f"<div style='width:{int(confidence*100)}%;height:20px;background-color:{confidence_color};'></div>", 
                            unsafe_allow_html=True)

        with test_col2:
            test_query = "Who is the latest Prime Minister of India?"
            st.text_input("Test Query", value=test_query, key="off_topic_query")
            if st.button("Test Off-Topic Query"):
                log("Running Test case with Low confidence query")
                with st.spinner("Testing..."):
                    if rag_type == RagTypes.basic:
                        answer, confidence, chunks, scores = st.session_state.basic_rag.answer(
                            test_query, st.session_state.texts, st.session_state.embeddings, top_k
                        )
                    else:
                        answer, confidence, chunks, scores = st.session_state.advanced_rag.answer(
                            test_query, st.session_state.texts, st.session_state.embeddings, top_k
                        )
                    st.session_state.test_res_low['answer'] = answer
                    st.session_state.test_res_low["confidence"] = confidence
                    st.session_state.test_res_low["chunks"] = chunks
                    st.session_state.test_res_low["scores"] = scores
            test_res_low = st.session_state.test_res_low
            if test_res_low:
                answer = test_res_low['answer']
                confidence = test_res_low['confidence']
                st.markdown(f"**Answer:** {answer}")
                st.markdown(f"### Confidence Score: {confidence:.2f}")
                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.markdown(f"<div style='width:{int(confidence*100)}%;height:20px;background-color:{confidence_color};'></div>", 
                            unsafe_allow_html=True)
    st.write("")

# Streamlit UI Implementations END ######################################################################
