from llama_cpp import Llama
from transformers import AutoTokenizer
import torch


def load_olmo_model(n_ctx=2048, n_threads=8):
    """
    Load the OLMo GGUF model and tokenizer.
    
    Args:
        n_ctx: Context window size
        n_threads: Number of CPU threads
        
    Returns:
        model: The loaded Llama model
        tokenizer: The loaded tokenizer
    """
    model = Llama.from_pretrained(
        repo_id="mradermacher/OLMo-1B-Base-shakespeare-GGUF",
        filename="OLMo-1B-Base-shakespeare.IQ3_M.gguf",
        n_ctx=n_ctx,
        n_threads=n_threads,
        embedding=True,
        verbose=False
    )
    
    # Load tokenizer from base model
    try:
        tokenizer = AutoTokenizer.from_pretrained("mradermacher/OLMo-1B-Base-shakespeare-GGUF", trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B", trust_remote_code=True)
    
    return model, tokenizer


def get_embeddings(text, model, tokenizer):
    """
    Get embeddings for a given text using the OLMo model.
    
    Args:
        text: Input text string
        model: The loaded Llama model
        tokenizer: The loaded tokenizer (not used for embedding, but kept for API consistency)
        
    Returns:
        hidden_states: Tensor containing the embeddings
    """
    if hasattr(model, 'embed'):
        # model.embed() expects a text string and handles tokenization internally
        embeddings = model.embed(text)
        hidden_states = torch.tensor(embeddings, dtype=torch.float32)
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)
    else:
        raise RuntimeError("Model does not support embedding extraction. Update llama-cpp-python.")
    
    return hidden_states


def get_mean_pooled_embeddings(text, model, tokenizer):
    """
    Get mean-pooled embeddings for a given text.
    
    Args:
        text: Input text string
        model: The loaded model
        tokenizer: The loaded tokenizer
        
    Returns:
        pooled_embedding: Mean-pooled embedding tensor
    """
    hidden_states = get_embeddings(text, model, tokenizer)
    pooled_embedding = torch.mean(hidden_states, dim=1)
    return pooled_embedding.squeeze(0)


def get_cls_embedding(text, model, tokenizer):
    """
    Get the first token embedding for a given text.
    
    Args:
        text: Input text string
        model: The loaded model
        tokenizer: The loaded tokenizer
        
    Returns:
        cls_embedding: First token embedding tensor
    """
    hidden_states = get_embeddings(text, model, tokenizer)
    cls_embedding = hidden_states[:, 0, :]
    return cls_embedding.squeeze(0)


def extract_entity_embeddings(entities_contexts, model, tokenizer, pooling_method="mean"):
    """
    Extract embeddings for entities from their contexts.
    
    Args:
        entities_contexts: Dictionary mapping entity names to lists of context strings
        model: The loaded model
        tokenizer: The loaded tokenizer
        pooling_method: "mean" for mean pooling, "cls" for first token, or "all" for all token embeddings
        
    Returns:
        entities_embeddings: Dictionary mapping entity names to lists of embeddings
    """
    entities_embeddings = {}
    
    for entity, contexts in entities_contexts.items():
        entities_embeddings[entity] = []
        
        for context in contexts:
            if pooling_method == "mean":
                embedding = get_mean_pooled_embeddings(context, model, tokenizer)
            elif pooling_method == "cls":
                embedding = get_cls_embedding(context, model, tokenizer)
            elif pooling_method == "all":
                embedding = get_embeddings(context, model, tokenizer)
                embedding = embedding.squeeze(0)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")
            
            entities_embeddings[entity].append(embedding)
    
    return entities_embeddings


def average_embeddings(entities_embeddings):
    """
    Average embeddings for each entity across all contexts.
    
    Args:
        entities_embeddings: Dictionary mapping entity names to lists of embedding tensors
        
    Returns:
        averaged_embeddings: Dictionary mapping entity names to averaged embedding tensors
    """
    averaged_embeddings = {}
    
    for entity, embeddings_list in entities_embeddings.items():
        if len(embeddings_list) > 0:
            stacked_embeddings = torch.stack(embeddings_list)
            averaged_embedding = torch.mean(stacked_embeddings, dim=0)
            averaged_embeddings[entity] = averaged_embedding
        else:
            print(f"Warning: No embeddings found for entity '{entity}'")
    
    return averaged_embeddings


# Example usage
if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_olmo_model()
    print("Model loaded successfully!")
    
    example_text = "To be or not to be, that is the question."
    print(f"\nGetting embeddings for: '{example_text}'")
    
    mean_embedding = get_mean_pooled_embeddings(example_text, model, tokenizer)
    print(f"Mean-pooled embedding shape: {mean_embedding.shape}")
    
    cls_embedding = get_cls_embedding(example_text, model, tokenizer)
    print(f"CLS embedding shape: {cls_embedding.shape}")
    
    all_embeddings = get_embeddings(example_text, model, tokenizer)
    print(f"All token embeddings shape: {all_embeddings.shape}")
