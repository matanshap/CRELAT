from transformers import BertTokenizer, BertModel
import torch
from scipy import spatial


class BERTModelManager:
    """
    Manages BERT model and tokenizer to avoid reloading them on every inference call.
    This significantly improves performance when processing multiple texts.
    """
    _instances = {}
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BERT model '{model_name}' on {self.device}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        print(f"BERT model loaded successfully!")
    
    @classmethod
    def get_instance(cls, model_name='bert-base-uncased'):
        """Get or create a singleton instance for the given model name."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]
    
    def get_embedding(self, text):
        """
        Get embedding vector for a single text using the cached BERT model.
        
        Args:
            text: A string containing the text
        
        Returns:
            embedding: A torch tensor containing the text embedding vector (mean pooling of all tokens)
        """
        # Tokenize the text
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            hidden_states = outputs[0]  # The hidden states from all layers
        
        # Compute mean pooling (average of all token embeddings)
        context_embedding = torch.mean(hidden_states[0], dim=0)  # Mean pooling over all tokens
        
        return context_embedding
    
    def get_embeddings_batch(self, texts, batch_size=32):
        """
        Get embeddings for multiple texts in batches for better performance.
        
        Args:
            texts: List of strings
            batch_size: Number of texts to process at once
        
        Returns:
            embeddings: List of torch tensors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**encoded)
                hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]
            
            # Compute mean pooling for each text in batch (ignoring padding tokens)
            attention_mask = encoded['attention_mask']  # [batch_size, seq_len]
            for j in range(len(batch_texts)):
                # Mask out padding tokens
                mask = attention_mask[j].unsqueeze(-1)  # [seq_len, 1]
                masked_hidden = hidden_states[j] * mask  # [seq_len, hidden_size]
                # Sum and divide by number of non-padding tokens
                sum_embeddings = torch.sum(masked_hidden, dim=0)
                num_tokens = torch.sum(attention_mask[j])
                embedding = sum_embeddings / num_tokens
                embeddings.append(embedding)
        
        return embeddings


def extract_entity_contexts(tokens, entities, context_window=10):
    # Tokenize the book text
    # Find the positions of each entity in the tokenized text
    lower_entities = [entity[0].lower() for entity in entities]
    entity_positions = dict()
    for index, token in enumerate(tokens):
        if token.lower() in lower_entities:
            if token.lower() in entity_positions.keys():
                entity_positions[token.lower()].append(index)
            else:
                entity_positions[token.lower()] = [index]

    # Extract contexts for each entity
    entity_contexts = dict()
    for entity, positions in entity_positions.items():
        last_position = -1
        for position in positions:
            # Ensure there's no overlap by starting the next context after the previous one
            start = max(position - context_window, last_position + 1)
            end = min(position + context_window + 1, len(tokens))
            context = tokens[start:end]
            if entity in entity_contexts:
                entity_contexts[entity].append(' '.join(context))
            else:
                entity_contexts[entity] = [' '.join(context)]
            last_position = position  # Update last position

    return entity_contexts

def inference_bert_single(context, model='bert-base-uncased', model_manager=None):
    """
    Get embedding vector for a single context using BERT.
    Uses cached model manager for better performance.
    
    Args:
        context: A string containing the context text
        model: Model name (default: 'bert-base-uncased')
        model_manager: Optional BERTModelManager instance to reuse (for performance)
    
    Returns:
        embedding: A torch tensor containing the context embedding vector (mean pooling of all tokens)
    """
    if model_manager is None:
        model_manager = BERTModelManager.get_instance(model)
    
    return model_manager.get_embedding(context)

def inference_bert(entities_contexts, model = 'bert-base-uncased'):
    # Step 1: Load pre-trained BERT model and tokenizer
    model_name = model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    entities_embeddings_per_context = dict()
    cls_per_context = dict()
    for entity in entities_contexts.keys():
        entities_embeddings_per_context[entity] = []
        cls_per_context[entity] = []
        contexts = entities_contexts[entity]
        for cont in contexts:
            # Step 2: Tokenize the context text with BERT
            text = cont
            target_entity = entity  # The entity we want to extract embedding for
            
            # Tokenize the full context
            tokenized_text = tokenizer.tokenize(text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])

            # Step 3: Find the entity position in BERT tokenized sequence
            # Tokenize the entity itself to get its subword tokens
            entity_tokens = tokenizer.tokenize(target_entity)
            
            # Find where the entity tokens appear in the tokenized context
            entity_token_indices = []
            for i in range(len(tokenized_text) - len(entity_tokens) + 1):
                if tokenized_text[i:i+len(entity_tokens)] == entity_tokens:
                    entity_token_indices = list(range(i, i+len(entity_tokens)))
                    break
            
            # If entity not found, try case-insensitive matching
            if not entity_token_indices:
                entity_lower_tokens = tokenizer.tokenize(target_entity.lower())
                for i in range(len(tokenized_text) - len(entity_lower_tokens) + 1):
                    if tokenized_text[i:i+len(entity_lower_tokens)] == entity_lower_tokens:
                        entity_token_indices = list(range(i, i+len(entity_lower_tokens)))
                        break

            # Step 4: Handle case where entity is not found
            if not entity_token_indices:
                print(f"Warning: Entity '{target_entity}' not found in context. Using CLS token embedding.")
                # Use CLS token (index 0) as fallback
                entity_token_indices = [0]
            
            # Step 5: Move the model and input tensors to the GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            tokens_tensor = tokens_tensor.to(device)

            # Step 6: Obtain the model's output
            with torch.no_grad():
                outputs = model(tokens_tensor)
                hidden_states = outputs[0]  # The hidden states from all layers

            # Step 7: Extract embedding for the entity (average if it spans multiple subword tokens)
            if len(entity_token_indices) == 1:
                entity_embedding = hidden_states[0, entity_token_indices[0]]
            else:
                # Average the embeddings of all subword tokens that make up the entity
                entity_embeddings = hidden_states[0, entity_token_indices]
                entity_embedding = torch.mean(entity_embeddings, dim=0)
            
            entities_embeddings_per_context[entity].append(entity_embedding)

            cls = outputs.last_hidden_state[:, 0, :]
            cls_per_context[entity].append(cls)
    return entities_embeddings_per_context, cls_per_context

def Gen_Bert_Pairs(entities_embeddings_per_context):
    all_pairs = [(a, b) for idx, a in enumerate(list(entities_embeddings_per_context.keys())) for b in list(entities_embeddings_per_context.keys())[idx + 1:]]
    for i in range(0,len(all_pairs)):
      all_pairs[i]=list(all_pairs[i])
    for idx,pair in enumerate(all_pairs):
        first_in_pair = pair[0]
        second_in_pair = pair[1]
        sim1 = 1 - spatial.distance.cosine(entities_embeddings_per_context[first_in_pair].cpu(), entities_embeddings_per_context[second_in_pair].cpu())
        all_pairs[idx].append(sim1)
    return all_pairs