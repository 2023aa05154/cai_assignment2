import os
import torch
from torch import cuda

# Fix for torch issue with streamlit package (ref: https://github.com/VikParuchuri/marker/issues/442)
torch.classes.__path__ = [
    os.path.join(torch.__path__[0], torch.classes.__file__)]


from transformers import AutoTokenizer, AutoModel, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import datetime


# Define constants
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
# Small, efficient model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Small LLM for generation
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# RAG Prompt
rag_prompt = """
Based on the following financial document excerpts, please answer the question.

Question: {query}

Context:
{context}


If above context is not relevant to answer the query, simply reply "This information is not available in the context"

Answer:"""


def log(text):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} : {__name__} : {text}")


class RagTypes:
    basic: str = "Basic RAG"
    advanced: str = "Advanced RAG"


# Embedding Model used by both the RAG approaches
class EmbeddingModel:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)

    def get_embeddings(self, texts):
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True,
            return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Mean Pooling - Take attention mask into account for averaging
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                model_output.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(
                model_output.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        return embeddings.cpu().numpy()


# Response generator used by both the RAG approaches
class ResponseGenerator:
    def __init__(self, model_name=LLM_MODEL_NAME):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device=0 if DEVICE == 'cuda' else -1,
            max_new_tokens=1024,
            return_full_text=False
        )

    def generate_response(self, query, context):
        prompt = rag_prompt.format(query=query, context=context)
        response = self.generator(
            prompt, do_sample=True, temperature=0.3
            )[0]['generated_text'].strip().replace('$', '\\$')
        # # Extract the answer part
        # if "Answer:" in response:
        #     answer = response.split("Answer:")[-1].strip()
        # else:
        #     answer = response.split(prompt)[-1].strip()

        return response


class Utils:
    response_generator = ResponseGenerator()
    embedding_model = EmbeddingModel()
