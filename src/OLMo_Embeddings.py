from llama_cpp import Llama
from transformers import AutoTokenizer
import torch


class OLMoModelManager:
    """
    Manages OLMo model and tokenizer to avoid reloading them on every inference call.
    This significantly improves performance when processing multiple texts.
    """
    _instances = {}
    
    def __init__(self, n_ctx=2048, n_threads=8, repo_id="mradermacher/OLMo-1B-Base-shakespeare-GGUF", 
                 filename="OLMo-1B-Base-shakespeare.IQ3_M.gguf"):
        """
        Initialize the OLMo model manager.
        
        Args:
            n_ctx: Context window size
            n_threads: Number of CPU threads
            repo_id: HuggingFace repository ID for the model
            filename: Model filename to load
        """
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.repo_id = repo_id
        self.filename = filename
        
        print(f"Loading OLMo model '{repo_id}/{filename}' (n_ctx={n_ctx}, n_threads={n_threads})...")
        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            embedding=True,
            verbose=False
        )
        
        # Load tokenizer from base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
        
        print(f"OLMo model loaded successfully!")
    
    @classmethod
    def get_instance(cls, n_ctx=2048, n_threads=8, repo_id="mradermacher/OLMo-1B-Base-shakespeare-GGUF",
                     filename="OLMo-1B-Base-shakespeare.IQ3_M.gguf"):
        """
        Get or create a singleton instance for the given configuration.
        
        Args:
            n_ctx: Context window size
            n_threads: Number of CPU threads
            repo_id: HuggingFace repository ID for the model
            filename: Model filename to load
            
        Returns:
            OLMoModelManager instance
        """
        # Create a unique key for the configuration
        config_key = (n_ctx, n_threads, repo_id, filename)
        if config_key not in cls._instances:
            cls._instances[config_key] = cls(n_ctx, n_threads, repo_id, filename)
        return cls._instances[config_key]
    
    def get_embeddings(self, text):
        """
        Get embeddings for a given text using the cached OLMo model.
        
        Args:
            text: Input text string
            
        Returns:
            hidden_states: Tensor containing the embeddings
        """
        if hasattr(self.model, 'embed'):
            # model.embed() expects a text string and handles tokenization internally
            embeddings = self.model.embed(text)
            hidden_states = torch.tensor(embeddings, dtype=torch.float32)
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states.unsqueeze(0)
        else:
            raise RuntimeError("Model does not support embedding extraction. Update llama-cpp-python.")
        
        return hidden_states
    
    def get_mean_pooled_embeddings(self, text):
        """
        Get mean-pooled embeddings for a given text using the cached model.
        
        Args:
            text: Input text string
            
        Returns:
            pooled_embedding: Mean-pooled embedding tensor
        """
        hidden_states = self.get_embeddings(text)
        pooled_embedding = torch.mean(hidden_states, dim=1)
        return pooled_embedding.squeeze(0)
    
    def get_cls_embedding(self, text):
        """
        Get the first token embedding for a given text using the cached model.
        
        Args:
            text: Input text string
            
        Returns:
            cls_embedding: First token embedding tensor
        """
        hidden_states = self.get_embeddings(text)
        cls_embedding = hidden_states[:, 0, :]
        return cls_embedding.squeeze(0)
    
    def extract_entity_embeddings(self, entities_contexts, pooling_method="mean"):
        """
        Extract embeddings for entities from their contexts using the cached model.
        
        Args:
            entities_contexts: Dictionary mapping entity names to lists of context strings
            pooling_method: "mean" for mean pooling, "cls" for first token, or "all" for all token embeddings
            
        Returns:
            entities_embeddings: Dictionary mapping entity names to lists of embeddings
        """
        entities_embeddings = {}
        
        for entity, contexts in entities_contexts.items():
            entities_embeddings[entity] = []
            
            for context in contexts:
                if pooling_method == "mean":
                    embedding = self.get_mean_pooled_embeddings(context)
                elif pooling_method == "cls":
                    embedding = self.get_cls_embedding(context)
                elif pooling_method == "all":
                    embedding = self.get_embeddings(context)
                    embedding = embedding.squeeze(0)
                else:
                    raise ValueError(f"Unknown pooling method: {pooling_method}")
                
                entities_embeddings[entity].append(embedding)
        
        return entities_embeddings


def get_embeddings_batch(
    texts,
    batch_size=32,
    n_ctx=2048,
    n_threads=8,
    repo_id="mradermacher/OLMo-1B-Base-shakespeare-GGUF",
    filename="OLMo-1B-Base-shakespeare.IQ3_M.gguf",
):
    """
    Get embeddings for multiple texts in batches for better performance.
    This function is independent of the OLMoModelManager class.
    
    Args:
        texts: List of strings
        batch_size: Number of texts to process at once
        n_ctx: Context window size
        n_threads: Number of CPU threads
        repo_id: HuggingFace repository ID for the GGUF model
        filename: GGUF filename within the HuggingFace repository
        
    Returns:
        embeddings: List of torch tensors (mean-pooled embeddings)
    """
    # Load the model (HuggingFace)
    model = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=n_ctx,
        n_threads=n_threads,
        embedding=True,
        verbose=False
    )
    
    if not hasattr(model, 'embed'):
        raise RuntimeError("Model does not support embedding extraction. Update llama-cpp-python.")
    
    embeddings = []
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Process each text in the batch
            for text in batch_texts:
                # Get embeddings from model
                embedding_data = model.embed(text)
                hidden_states = torch.tensor(embedding_data, dtype=torch.float32)
                
                # Ensure correct shape (batch_size, seq_len, hidden_dim)
                if len(hidden_states.shape) == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                
                # Mean pooling
                pooled_embedding = torch.mean(hidden_states, dim=1)
                embeddings.append(pooled_embedding.squeeze(0))
    finally:
        # Clean up model resources
        del model
    
    return embeddings
