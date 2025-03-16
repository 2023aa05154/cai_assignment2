import os
import torch
from torch import cuda

# Fix for torch issue with streamlit package (ref: https://github.com/VikParuchuri/marker/issues/442)
torch.classes.__path__ = [
    os.path.join(torch.__path__[0], torch.classes.__file__)]


from transformers import AutoTokenizer, AutoModel, pipeline
import datetime


# Define constants
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
# Small, efficient model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Small LLM for generation
LLM_MODEL_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# RAG Prompt
rag_prompt = """<|system|>
You are a helpful assistant that answers questions about financial documents. 
Only provide information that is directly supported by the given context.
If you don't know the answer based on the context, say so clearly.
</s>

<|user|>
Based on the following financial document excerpts, please answer the question:

Question: {query}

Context:
{context}
</s>

<|assistant|>"""


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


class ResponseGenerator:
    def __init__(self, model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        """
        Initialize a response generator using a GGUF quantized model for CPU efficiency.

        Args:
            model_path: Path to the GGUF model file. Default is TinyLlama 1.1B in 4-bit quantization.
        """
        try:
            from llama_cpp import Llama

            # Check if model exists, if not, download it
            if not os.path.exists(model_path):
                self._download_model(model_path)

            # Initialize the model with CPU-friendly settings
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,           # Context window size
                n_threads=4,          # Number of CPU threads to use
                n_batch=512,          # Batch size for prompt processing
                verbose=False         # Set to True for debugging
            )
            self.loaded = True

        except ImportError:
            log("llama-cpp-python package not installed")
            self.loaded = False

    def _download_model(self, model_path):
        """Download the GGUF model file if not present."""
        import requests
        import os

        # Map of model filenames to their Hugging Face repo URLs
        model_urls = {
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": 
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "tinyllama-1.1b-chat-v1.0.Q8_0.gguf": 
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        }

        # Get the URL for the specified model
        if model_path in model_urls:
            url = model_urls[model_path]
        else:
            log(f"No download URL found for {model_path}")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

        # Download with progress bar
        response = requests.get(url, stream=True)
        block_size = 1024  # 1 Kibibyte

        with open(model_path, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)

        log(f"Downloaded {model_path} successfully")

    def generate_response(self, query, context):
        """
        Generate a response based on the query and context using the quantized LLM.

        Args:
            query: The user's question
            context: The retrieved document chunks as context

        Returns:
            A generated answer string
        """
        if not self.loaded:
            return "Model could not be loaded. Please check your installation."

        # Format prompt using the chat template compatible with TinyLlama chat models
        prompt = rag_prompt.format(context=context, query=query)

        # Generate response
        response = self.llm(
            prompt,
            max_tokens=512,          # Maximum new tokens to generate
            temperature=0.3,         # Lower is more deterministic
            top_p=0.9,               # Nucleus sampling
            repeat_penalty=1.1,      # Penalty for repeating tokens
            echo=False               # Don't echo the prompt in the output
        )

        # Extract answer text
        answer = response['choices'][0]['text'].strip()

        return answer

    def get_model_info(self):
        """Return information about the loaded model."""
        if not self.loaded:
            return "Model not loaded"

        return {
            "type": "GGUF quantized LLM",
            "n_ctx": self.llm.n_ctx,
            "n_threads": self.llm.n_threads
        }


class Utils:
    response_generator = ResponseGenerator()
    embedding_model = EmbeddingModel()
