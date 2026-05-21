from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import os
import sys
from scipy import spatial
from huggingface_hub import snapshot_download


class BERTModelManager:
    """
    Manages BERT model and tokenizer to avoid reloading them on every inference call.
    This significantly improves performance when processing multiple texts.
    """
    _instances = {}
    
    def _load_tiny_gpt_shakespeare(self):
        """Custom loader for the bmeyer2025/tiny-gpt-shakespeare model."""
        repo_id = "bmeyer2025/tiny-gpt-shakespeare"
        local_dir = os.path.join(os.getcwd(), "models", "tiny-gpt-shakespeare")
        
        print(f"Downloading custom TinyGPT model from {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        
        # Add src to path for custom model imports
        src_path = os.path.join(local_dir, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        try:
            from model_modern import ModernGPT
        except ImportError:
            # Fallback for different directory structures if any
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_modern", os.path.join(src_path, "model_modern.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ModernGPT = mod.ModernGPT
            
        # Load checkpoint
        ckpt_path = os.path.join(local_dir, "model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Instantiate model
        raw_model = ModernGPT(**ckpt["config"])
        raw_model.load_state_dict(ckpt["model_state"])
        
        # Setup tokenizer (it uses GPT-2 vocab)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define wrapper to extract embeddings and match transformers API
        class TinyGPTExtractor(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.captured_emb = None
                self._hook_handle = self.model.ln_f.register_forward_hook(self._hook)
                
            def _hook(self, module, input, output):
                self.captured_emb = output
                
            def forward(self, input_ids=None, idx=None, attention_mask=None, **kwargs):
                self.captured_emb = None
                # Use input_ids if provided (transformers style) or idx (ModernGPT style)
                tokens = input_ids if input_ids is not None else idx
                # ModernGPT forward signature: (idx, targets=None, use_cache=False)
                self.model(tokens)
                
                # Create a ModelOutput-like object
                class ModelOutput:
                    def __init__(self, last_hidden_state):
                        self.last_hidden_state = last_hidden_state
                    def __getitem__(self, idx):
                        if idx == 0: return self.last_hidden_state
                        raise IndexError
                
                return ModelOutput(self.captured_emb)
                
        self.model = TinyGPTExtractor(raw_model)
        return True
    
    def _load_cyclicformer(self, repo_id):
        """Custom loader for CyclicFormer models."""
        local_dir = os.path.join(os.getcwd(), "models", "cyclicformer_checkpoint")
        print(f"Downloading CyclicFormer model from {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        
        # Add our local implementation path to sys.path
        impl_path = os.path.join(os.getcwd(), "models", "cyclicformer")
        if impl_path not in sys.path:
            sys.path.insert(0, impl_path)
            
        # Force reload of custom modules to avoid stale cache
        for mod in ["modeling_cyclicformer", "configuration_cyclicformer"]:
            if mod in sys.modules:
                del sys.modules[mod]
                
        from modeling_cyclicformer import CyclicFormerForCausalLM
        from configuration_cyclicformer import CyclicFormerConfig
        
        # Load config and model
        config = CyclicFormerConfig.from_pretrained(repo_id)
        raw_model = CyclicFormerForCausalLM(config)
        
        # Load weights (safetensors)
        from safetensors.torch import load_file
        weights_path = os.path.join(local_dir, "model.safetensors")
        state_dict = load_file(weights_path)
        raw_model.load_state_dict(state_dict)
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # The model's forward returns CausalLMOutputWithPast, but we want hidden states.
        # We can use the underlying CyclicFormerModel which returns BaseModelOutputWithPast.
        class CyclicExtractor(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.model # The CyclicFormerModel
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return self.model(input_ids=input_ids, attention_mask=attention_mask)
                
        self.model = CyclicExtractor(raw_model)
        return True
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading transformer model '{model_name}' on {self.device}...")
        
        repo_id = model_name
        subfolder = None
        
        # Handle cases like "namespace/repo/checkpoint-123" for HF hub
        if not os.path.isdir(model_name) and model_name.count('/') > 1:
            parts = model_name.split('/')
            # Typical HF repo is "user/repo" (2 parts)
            repo_id = "/".join(parts[:2])
            subfolder = "/".join(parts[2:])
            print(f"Interpreting '{model_name}' as repo '{repo_id}' with subfolder '{subfolder}'")
        
        # Special case for custom tiny-gpt model
        if repo_id == "bmeyer2025/tiny-gpt-shakespeare":
            try:
                if self._load_tiny_gpt_shakespeare():
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"Model '{model_name}' loaded (cached for reuse).")
                    return
            except Exception as e:
                print(f"Error loading custom TinyGPT model: {e}")
                if "CUDA" in str(e) or "device" in str(e):
                    print("Attempting to load on CPU instead...")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"Model '{model_name}' loaded on CPU.")
                    return
                raise e

        # Special case for CyclicFormer
        if repo_id == "Q-bert/CyclicFormer-tiny-shakespeare":
            try:
                if self._load_cyclicformer(repo_id):
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"Model '{model_name}' loaded (cached for reuse).")
                    return
            except Exception as e:
                print(f"Error loading CyclicFormer model: {e}")
                if "CUDA" in str(e) or "device" in str(e):
                    print("Attempting to load on CPU instead...")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"Model '{model_name}' loaded on CPU.")
                    return
                raise e

        # Try loading as PEFT model if peft is available
        loaded_peft = False
        self._added_pad_token = False
        
        # Only pass subfolder if it's explicitly provided to avoid TypeError in some transformers versions
        kwargs = {"subfolder": subfolder} if subfolder else {}
        
        try:
            from peft import PeftModel, PeftConfig
            # Attempt to load config to see if it's a PEFT model
            peft_config = PeftConfig.from_pretrained(repo_id, **kwargs)
            base_model_name = peft_config.base_model_name_or_path
            print(f"Detected PEFT model. Loading base model '{base_model_name}'...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self._setup_pad_token()
            
            base_model = AutoModel.from_pretrained(base_model_name)
            if self._added_pad_token:
                base_model.resize_token_embeddings(len(self.tokenizer))
            
            # Use generic PeftModel class instead of PeftModel.from_pretrained to avoid
            # task-specific wrappers (like QA) that can cause signature mismatches.
            self.model = PeftModel(base_model, peft_config)
            self.model.load_adapter(repo_id, "default", **kwargs)
            loaded_peft = True
            print(f"Loaded PEFT adapter from {repo_id}")
        except Exception as e:
            # Fallback to standard loading if peft is missing or it's not a PEFT model
            pass
            
        if not loaded_peft:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
            self._setup_pad_token()
            try:
                try:
                    self.model = AutoModel.from_pretrained(repo_id, **kwargs)
                except ValueError as e:
                    # Handle cases where AutoModel fails because it's a Seq2Seq model
                    if "EncoderDecoderConfig" in str(e) or "Seq2Seq" in str(e):
                        print(f"Detected Seq2Seq model, loading with AutoModelForSeq2SeqLM...")
                        raw_model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, **kwargs)
                        self.model = Seq2SeqExtractor(raw_model)
                    else:
                        raise e
                
                if self._added_pad_token and not isinstance(self.model, Seq2SeqExtractor):
                    self.model.resize_token_embeddings(len(self.tokenizer))
                elif self._added_pad_token and isinstance(self.model, Seq2SeqExtractor):
                    self.model.model.resize_token_embeddings(len(self.tokenizer))
                    
                self.model.to(self.device)
            except Exception as e:
                if "CUDA" in str(e) or "device" in str(e):
                    print(f"CUDA error during loading: {e}. Falling back to CPU...")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                else:
                    raise e
        
        self.model.eval()
        print(f"Model '{model_name}' loaded (cached for reuse).")

    def _setup_pad_token(self):
        """Ensure the tokenizer has a pad token."""
        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif getattr(self.tokenizer, "unk_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self._added_pad_token = True
    
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
            
            # Determine max_length based on model config if available
            config = getattr(self.model, "config", None)
            if config is not None and not isinstance(config, dict):
                # HuggingFace config objects
                max_length = getattr(config, "max_position_embeddings", 512)
            elif isinstance(config, dict):
                max_length = config.get("max_position_embeddings", 512)
            else:
                max_length = 512
                
            # Special check for our custom TinyGPT wrapper which has block_size
            if hasattr(self.model, "model") and hasattr(self.model.model, "block_size"):
                max_length = self.model.model.block_size
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get model output
            try:
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                # Print more info about the inputs
                print(f"Input ids shape: {encoded['input_ids'].shape}")
                raise e
            
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


class Seq2SeqExtractor(torch.nn.Module):
    """Wrapper for Seq2Seq models to return encoder hidden states."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # For Seq2Seq models, we only care about the encoder's output for embeddings
        outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        # Return (last_hidden_state,) to match BERT style (outputs[0])
        return (outputs.last_hidden_state,)

def get_embeddings_batch(texts, model_name='bert-base-uncased', batch_size=32):
    """
    Mean-pooled token embeddings per string. Uses ``BERTModelManager.get_instance`` so each
    ``model_name`` is loaded at most once per process.
    """
    manager = BERTModelManager.get_instance(model_name)
    return manager.get_embeddings_batch(texts, batch_size=batch_size)


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