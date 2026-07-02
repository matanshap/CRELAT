from llama_cpp import Llama
import torch

def get_embeddings_batch(repo_id, filename, texts, batch_size=32, n_ctx=2048, n_threads=8):
    """
    Get embeddings for multiple texts in batches for better performance.
    This function is independent of the OLMoModelManager class.
    
    Args: 
        model_path: Path to the GGUF model file
        texts: List of strings
        batch_size: Number of texts to process at once
        n_ctx: Context window size
        n_threads: Number of CPU threads
        
    Returns:
        embeddings: List of torch tensors (mean-pooled embeddings)
    """
    # Load the model
    print(f"Loading GGUF model '{repo_id}/{filename}' (n_ctx={n_ctx}, n_threads={n_threads})...")
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
