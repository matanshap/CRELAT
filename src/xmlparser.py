import xml.etree.ElementTree as ET
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from BERT_Inference_Without_Finetune import get_embeddings_batch as get_embeddings_transformers
from OLMo_Embeddings import get_embeddings_batch as get_embeddings_olmo
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model registry – maps short names to HuggingFace identifiers + backend.
# "transformers" backend → uses AutoModel/AutoTokenizer (any HF model).
# "olmo"          backend → uses llama_cpp via OLMo_Embeddings (GGUF models).
#
# Any HuggingFace model name that is NOT in the registry is automatically
# treated as a "transformers" backend model, so you can pass e.g.
# "sentence-transformers/all-MiniLM-L6-v2" directly.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "bert": {
        "hf_name": "bert-base-uncased",
        "backend": "transformers",
    },
    "macberth": {
        "hf_name": "emanjavacas/MacBERTh",
        "backend": "transformers",
    },
    "olmo": {
        "hf_name": "mradermacher/OLMo-1B-Base-shakespeare-GGUF",
        "backend": "olmo",
        "olmo_filename": "OLMo-1B-Base-shakespeare.IQ3_M.gguf",
        "olmo_n_ctx": 2048,
        "olmo_n_threads": 8,
    },
}


def slugify_transformer_model(model_name):
    """Stable suffix for speech embedding keys: average_<slug>_embedding."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (model_name or "").strip()).strip("_").lower()
    return s or "model"


def resolve_model(name: str) -> tuple[str, str, dict]:
    """Return (slug, backend, config) for a model short-name or HF identifier.

    * Known short names are looked up in MODEL_REGISTRY.
    * Unknown names are assumed to be HuggingFace transformer model IDs.
    """
    if name in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[name]
        return name, entry["backend"], entry
    slug = slugify_transformer_model(name)
    return slug, "transformers", {"hf_name": name, "backend": "transformers"}


def read_tei_play_title(xml_path):
    """
    Folger TEI: first ``<title>`` under ``teiHeader/fileDesc/titleStmt`` (e.g. Coriolanus).
    Returns None if missing or unreadable.
    """
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return None
    header = root.find("tei:teiHeader", ns)
    if header is None:
        return None
    fd = header.find("tei:fileDesc", ns)
    if fd is None:
        return None
    ts = fd.find("tei:titleStmt", ns)
    if ts is None:
        return None
    for title_el in ts.findall("tei:title", ns):
        t = (title_el.text or "").strip()
        if t:
            return t
    return None


class XMLParser:
    namespaces = {
        'tei': 'http://www.tei-c.org/ns/1.0',
        "xml": 'http://www.w3.org/XML/1998/namespace',
    }

    def __init__(
        self,
        xml_path,
        options={"co-oc", "bert"},
        model=None,
        embedding_batch_size=16,
        # Deprecated – kept for backward compat; prefer ``model``.
        transformer_model_name=None,
    ):
        """
        Parameters
        ----------
        xml_path : str
            Path to a Folger TEI XML play file.
        options : set[str]
            Toggles.  ``"co-oc"`` enables co-occurrence counts.
            Any model short-name (``"bert"``, ``"olmo"``, ``"macberth"``)
            or full HuggingFace identifier (e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``) in the set will
            compute embeddings + cosine similarities with that model.
            ``"w2v"`` enables Word2Vec similarity.
        model : str, optional
            Convenience shorthand – equivalent to adding the name to
            *options*.  Accepts any key from ``MODEL_REGISTRY`` or any
            HuggingFace model name.
        embedding_batch_size : int
            Batch size for transformer inference.
        """
        self.xml_path = xml_path
        self.options = set(options)
        if model is not None:
            self.options.add(model)
        if transformer_model_name is not None:
            self.options.add(transformer_model_name)
        self.embedding_batch_size = embedding_batch_size
        self.text = None
        self.entities = None

        # Resolve which embedding models to run.
        self._model_configs: dict[str, tuple[str, dict]] = {}
        non_model_opts = {"co-oc", "w2v"}
        for opt in self.options:
            if opt in non_model_opts:
                continue
            slug, backend, cfg = resolve_model(opt)
            self._model_configs[slug] = (backend, cfg)

        # Cosine-similarity dicts keyed by model slug, populated by parse().
        self.cosine_similarities: dict[str, dict] = {}
        self.speech_interactions = []
        self.play_title = read_tei_play_title(self.xml_path) or os.path.basename(self.xml_path)
    
    def parse(self):
        with open(self.xml_path, 'r') as file:
            self.xml = file.read()
            self.root = ET.fromstring(self.xml)

        self.characters = [*self.__get_characters(), '[UNKNOWN]']
        self.characters_speeches = self.__get_characters_speeches()

        if "co-oc" in self.options:
            self.co_occurrences = self.__calculate_co_occurrences()

        for slug in self._model_configs:
            self.cosine_similarities[slug] = self.__calculate_cosine_similarity(
                embedding_type=slug)

        # Backward-compat aliases
        if "bert" in self.cosine_similarities:
            self.cosine_similarity_bert = self.cosine_similarities["bert"]
        if "olmo" in self.cosine_similarities:
            self.cosine_similarity_olmo = self.cosine_similarities["olmo"]

        if "w2v" in self.options:
            self.cosine_similarity_w2v = self.__calculate_w2v_similarity()

    def __get_raw_text_path(self, raw_text_dir='Data/raw_text'):
        """
        Resolve path to raw text file from Folger. Download if missing.
        Raises if Folger download fails (no fallback).
        """
        play_code = os.path.splitext(os.path.basename(self.xml_path))[0]
        from download_folger_raw_text import ensure_raw_text_exists
        return ensure_raw_text_exists(play_code, output_dir=raw_text_dir)

    def __load_raw_text_tokens(self, raw_text_path):
        """Load and tokenize raw play text for W2V training."""
        with open(raw_text_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        return re.findall(r"\b\w+\b", text.lower())

    def __calculate_w2v_similarity(
        self, vector_size=100, window_size=10, output_dir='output/',
        raw_text_dir='Data/raw_text'
    ):
        """
        Train W2V on the raw text of the play (download if missing) and compute
        cosine similarity between mean-centered character-name vectors.
        Returns dict of dicts matching the structure of cosine_similarity_bert.
        """
        from W2V import TrainW2VModel
        from scipy import spatial
        import numpy as np

        raw_text_path = self.__get_raw_text_path(raw_text_dir)
        if not raw_text_path or not os.path.exists(raw_text_path):
            code = os.path.splitext(os.path.basename(self.xml_path))[0]
            raise FileNotFoundError(
                f"Raw text from Folger not found for {code}. "
                f"Run: conda activate mind-the-gap && python src/download_folger_raw_text.py {code}"
            )

        self._w2v_params = {
            'vector_size': vector_size,
            'window_size': window_size,
        }

        corpus_list = self.__load_raw_text_tokens(raw_text_path)
        if len(corpus_list) < 10:
            return {k: {k: 0 for k in self.characters} for k in self.characters}

        book_name = os.path.splitext(os.path.basename(self.xml_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir if output_dir.endswith('/') else output_dir + '/'
        w2v_vectors = TrainW2VModel(
            book_name, corpus_list, vector_size, window_size, output_path, forward_only=False
        )
        vectors = w2v_vectors["vectors"]

        # Character embedding = W2V vector of the character NAME (as token in raw text)
        # Map XML ids (Hamlet_Ham) -> raw-text token (hamlet). Raw text has BARNARDO, HAMLET etc.
        def char_id_to_token(char_id):
            if char_id == '[UNKNOWN]':
                return 'unknown'
            part = char_id.split('_')[0].split('.')[-1]
            return part.lower()

        if vectors:
            target_dim = max(len(v) for v in vectors.values() if hasattr(v, '__len__'))
        else:
            target_dim = vector_size
        char_embeddings = {}
        for char in self.characters:
            token = char_id_to_token(char)
            if token in vectors:
                vec = np.array(vectors[token])
            else:
                vec = np.zeros(target_dim)
            if vec.size != target_dim:
                if vec.size == 0:
                    vec = np.zeros(target_dim)
                elif vec.size > target_dim:
                    vec = vec[:target_dim]
                else:
                    vec = np.pad(vec, (0, target_dim - vec.size), mode='constant')
            char_embeddings[char] = vec

        # Mean-center character name vectors before cosine similarity
        mean_chars = [c for c in self.characters if c != '[UNKNOWN]']
        if not mean_chars:
            mean_chars = list(self.characters)
        mean_matrix = np.vstack([char_embeddings[c] for c in mean_chars])
        mean_vec = mean_matrix.mean(axis=0)
        for char in char_embeddings:
            char_embeddings[char] = char_embeddings[char] - mean_vec

        # Pairwise cosine similarity
        cosine_similarities = {k: {k: 0 for k in self.characters} for k in self.characters}
        chars_list = list(self.characters)
        for i, c1 in enumerate(chars_list):
            for c2 in chars_list[i + 1:]:
                v1, v2 = char_embeddings[c1], char_embeddings[c2]
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sim = 1 - spatial.distance.cosine(v1, v2)
                else:
                    sim = 0.0
                cosine_similarities[c1][c2] = sim
                cosine_similarities[c2][c1] = sim
        return cosine_similarities

    def find_tag_occurrences(self, tag_name):
        """Find all elements with a specific tag name"""
        # Handle namespace
        results = self.root.findall(f'.//tei:{tag_name}', XMLParser.namespaces)
        return results
        
    def _normalize_speaker_id(self, speaker_id):
        if not speaker_id or speaker_id == '[UNKNOWN]':
            return speaker_id
        
        # Current play code from filename (e.g., R3, 1H6, Ham)
        play_code = os.path.splitext(os.path.basename(self.xml_path))[0]
        
        # If the ID has a suffix, replace it with the current play code.
        # Otherwise, append the current play code.
        if '_' in speaker_id:
            base = speaker_id.rsplit('_', 1)[0]
            return f"{base}_{play_code}"
        
        return f"{speaker_id}_{play_code}"

    def __get_characters(self):
        persons = self.find_tag_occurrences('person')
        persons.extend(self.find_tag_occurrences('personGrp'))
        characters = []
        for person in persons:
            raw_id = person.get(f'{{{XMLParser.namespaces["xml"]}}}id')
            characters.append(self._normalize_speaker_id(raw_id))
        return characters

    def __get_characters_speeches(self):
        """
        Returns list(list(dict))
        Each list(dict) contains the speeches of the characters in the scene
        Each dict contains the speaker's name and the speech text
        The list(list(dict)) contains all the scenes in the play
        """

        characters_speeches = []

        def _compute_embeddings(slug, backend, cfg, texts):
            """Run inference for one model and return a list of embedding tensors."""
            if backend == "olmo":
                return get_embeddings_olmo(
                    texts=texts,
                    repo_id=cfg.get("hf_name",
                                    "mradermacher/OLMo-1B-Base-shakespeare-GGUF"),
                    filename=cfg.get("olmo_filename",
                                     "OLMo-1B-Base-shakespeare.IQ3_M.gguf"),
                    n_ctx=cfg.get("olmo_n_ctx", 2048),
                    n_threads=cfg.get("olmo_n_threads", 8),
                    batch_size=self.embedding_batch_size,
                )
            # Default: HuggingFace transformers backend
            return get_embeddings_transformers(
                texts,
                batch_size=self.embedding_batch_size,
                model_name=cfg["hf_name"],
            )

        def _store_embeddings(slug, embeddings):
            idx = 0
            for scene in characters_speeches:
                for speech_info in scene:
                    speech_info[f'average_{slug}_embedding'] = embeddings[idx]
                    idx += 1

        scenes = self.find_tag_occurrences('div2')
        
        def _extract_folger_text(elem):
            res = ""
            for child in elem:
                if child.tag.endswith('speaker') or child.tag.endswith('stage') or child.tag.endswith('sound'):
                    continue
                if child.tag.endswith('lb'):
                    res += '\n'
                elif child.tag.endswith('w') or child.tag.endswith('c') or child.tag.endswith('pc'):
                    if child.text:
                        res += child.text
                res += _extract_folger_text(child)
            return res

        # Collect all speeches first
        for scene in scenes:
            scene_speeches = []
            for speech in scene.findall('tei:sp', XMLParser.namespaces):
                who = speech.get('who')
                raw_speaker = who.split()[0][1:] if who is not None else '[UNKNOWN]'
                speech_info = {
                    'speaker': self._normalize_speaker_id(raw_speaker),
                    'text': _extract_folger_text(speech).strip(),
                }
                scene_speeches.append(speech_info)
            characters_speeches.append(scene_speeches)
            
        speeches_texts = [speech['text'] for scene in characters_speeches for speech in scene]

        for slug, (backend, cfg) in self._model_configs.items():
            embeddings = _compute_embeddings(slug, backend, cfg, speeches_texts)
            _store_embeddings(slug, embeddings)

        return characters_speeches

    def __generate_speech_pairs(self):
        """
        Returns dict of speech pairs
        Each pair is a tuple of two characters
        Each value is a list of tuples (speech of char1, speech of char2)
        """
        speech_pairs = {(char1, char2): [] for char1 in self.characters for char2 in self.characters if char1 != char2}
        
        # Iterate through each scene
        for scene in self.characters_speeches:
            # Look for consecutive speeches between different characters
            for speech_idx in range(len(scene) - 1):
                speaker1 = scene[speech_idx]['speaker']
                speech1 = scene[speech_idx]['speech']
                
                speaker2 = scene[speech_idx + 1]['speaker']
                speech2 = scene[speech_idx + 1]['speech']
                
                # Only add pairs if the speakers are different and both are valid characters
                if speaker1 != speaker2 and speaker1 in self.characters and speaker2 in self.characters:
                    speech_pairs[(speaker1, speaker2)].append((speech1, speech2))
        
        return speech_pairs
        
    def __calculate_co_occurrences(self):

        co_occurrences = {k: {k: 0 for k in self.characters} for k in self.characters}
        for scene in self.characters_speeches:

            for speech_idx in range(len(scene) - 1):
                speaker = scene[speech_idx]['speaker']
                # speech = scene[speech_idx]['speech']

                next_speaker = scene[speech_idx + 1]['speaker']
                # next_speech = scene[speech_idx + 1]['speech']

                # TODO: Check if there is a character that speaks twice in a row
                if speaker != next_speaker:
                    co_occurrences[speaker][next_speaker] += 1
                    co_occurrences[next_speaker][speaker] += 1
                
        return co_occurrences

    def __calculate_scene_copresence(self, min_speeches_per_scene=1, include_unknown=False):
        """
        Count how often two characters appear in the same scene (co-presence).
        """
        copresence = {k: {k: 0 for k in self.characters} for k in self.characters}
        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']

        for scene in self.characters_speeches:
            present = set()
            for char in characters:
                count = sum(1 for sp in scene if sp.get('speaker') == char)
                if count >= min_speeches_per_scene:
                    present.add(char)
            present_list = sorted(present)
            for i, c1 in enumerate(present_list):
                for c2 in present_list[i + 1:]:
                    copresence[c1][c2] += 1
                    copresence[c2][c1] += 1

        return copresence

    def __calculate_cosine_similarity(self, embedding_type='bert'):
        cosine_similarities = {k: {k: 0 for k in self.characters} for k in self.characters}
        embedding_key = f'average_{embedding_type}_embedding'
        scene_labels = self.__get_scene_labels()

        for scene_idx, scene in enumerate(self.characters_speeches):
            scene_label = scene_labels[scene_idx] if scene_idx < len(scene_labels) else f"Scene {scene_idx+1}"
            for speech_idx in range(len(scene) - 1):
                speaker = scene[speech_idx]['speaker']
                next_speaker = scene[speech_idx + 1]['speaker']

                if speaker != next_speaker:
                    # If embeddings are 1D tensors, add a dimension for batch processing
                    sim = F.cosine_similarity(
                        scene[speech_idx][embedding_key].unsqueeze(0),
                        scene[speech_idx + 1][embedding_key].unsqueeze(0)
                    ).item()
                    
                    # Guard against NaN (e.g. from zero-vector embeddings)
                    if math.isnan(sim):
                        continue
                    
                    cosine_similarities[speaker][next_speaker] += sim
                    cosine_similarities[next_speaker][speaker] = cosine_similarities[speaker][next_speaker]

                    self.speech_interactions.append({
                        'play': self.play_title,
                        'scene': scene_label,
                        'speaker1': speaker,
                        'speaker2': next_speaker,
                        'text1': scene[speech_idx]['text'],
                        'text2': scene[speech_idx + 1]['text'],
                        'cosine_similarity': sim,
                        'model': embedding_type
                    })

        return cosine_similarities

    def __get_scene_labels(self):
        """
        Build scene labels in document order. If act info exists, label as A{act}.S{scene}.
        """
        if not hasattr(self, 'root') or self.root is None:
            return []

        labels = []
        div1s = self.root.findall('.//tei:div1', XMLParser.namespaces)
        if div1s:
            for act_idx, act in enumerate(div1s, start=1):
                scenes = act.findall('.//tei:div2', XMLParser.namespaces)
                for scene_idx, _ in enumerate(scenes, start=1):
                    labels.append(f"A{act_idx}.S{scene_idx}")
        else:
            scenes = self.root.findall('.//tei:div2', XMLParser.namespaces)
            for scene_idx, _ in enumerate(scenes, start=1):
                labels.append(f"Scene {scene_idx}")
        return labels

    def __compute_scene_character_centroids(self, embedding_type='bert'):
        """
        Compute per-scene, per-character centroid embeddings.
        Returns list of dicts aligned with self.characters_speeches order.
        """
        embedding_key = f'average_{embedding_type}_embedding'
        scene_centroids = []

        for scene in self.characters_speeches:
            char_vectors = {char: [] for char in self.characters}
            for speech in scene:
                speaker = speech.get('speaker', '[UNKNOWN]')
                if speaker not in char_vectors:
                    continue
                vec = speech.get(embedding_key)
                if vec is None:
                    continue
                try:
                    if hasattr(vec, 'detach'):
                        vec = vec.detach().cpu().numpy()
                except Exception:
                    pass
                char_vectors[speaker].append(np.array(vec))

            centroids = {}
            for char, vecs in char_vectors.items():
                if not vecs:
                    centroids[char] = None
                else:
                    centroids[char] = np.mean(np.vstack(vecs), axis=0)
            scene_centroids.append(centroids)

        return scene_centroids

    def __compute_character_centroids(self, embedding_type='bert'):
        """
        Compute per-character centroid embeddings by averaging all speech embeddings.
        """
        embedding_key = f'average_{embedding_type}_embedding'
        char_vectors = {char: [] for char in self.characters}

        for scene in self.characters_speeches:
            for speech in scene:
                speaker = speech.get('speaker', '[UNKNOWN]')
                if speaker not in char_vectors:
                    continue
                vec = speech.get(embedding_key)
                if vec is None:
                    continue
                try:
                    # torch.Tensor -> numpy
                    if hasattr(vec, 'detach'):
                        vec = vec.detach().cpu().numpy()
                except Exception:
                    pass
                char_vectors[speaker].append(np.array(vec))

        centroids = {}
        for char, vecs in char_vectors.items():
            if not vecs:
                centroids[char] = None
            else:
                centroids[char] = np.mean(np.vstack(vecs), axis=0)
        return centroids

    def _compute_centroid_similarity(self, embedding_type='bert'):
        """
        Build cosine similarity dict from per-character centroids.
        """
        from scipy import spatial
        centroids = self.__compute_character_centroids(embedding_type=embedding_type)
        cosine_similarities = {k: {k: 0 for k in self.characters} for k in self.characters}
        chars_list = list(self.characters)
        for i, c1 in enumerate(chars_list):
            for c2 in chars_list[i + 1:]:
                v1, v2 = centroids.get(c1), centroids.get(c2)
                if v1 is None or v2 is None:
                    sim = 0.0
                else:
                    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        sim = 1 - spatial.distance.cosine(v1, v2)
                    else:
                        sim = 0.0
                cosine_similarities[c1][c2] = sim
                cosine_similarities[c2][c1] = sim
        return cosine_similarities

    def _plot_interactions_scatter(
        self,
        cosine_similarity_dict,
        play_name,
        y_label,
        filename_suffix,
        output_dir='output/',
        characters_filter=None,
        use_y_ratio=False,
        use_softmax=False,
        show_trend=True,
        min_cooc_threshold=0,
        footer_text=None,
        left_panel_text=None,
        output_path=None,
    ):
        """
        Internal helper: scatter plot with X = interactions (co-occurrence) per pair,
        Y = cosine similarity per pair (or cosine/interactions if use_y_ratio=True).
        Pairs with co_occurrence < min_cooc_threshold are excluded.
        If output_path is set, the figure is saved there (dirs created); otherwise
        output_dir/{sanitized_play_name}_{filename_suffix}.svg.
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for interactions scatter plot")
        characters = list(self.co_occurrences.keys())
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        pairs_data = []
        for i, char1 in enumerate(characters):
            for char2 in characters[i + 1:]:
                cooc = self.co_occurrences[char1][char2]
                if cooc < min_cooc_threshold:
                    continue
                cosim = cosine_similarity_dict[char1][char2]
                if use_y_ratio and cooc == 0:
                    continue
                y_val = (cosim / cooc) if use_y_ratio else cosim
                pairs_data.append({
                    'char1': char1, 'char2': char2,
                    'co_occurrence': cooc, 'cosine_similarity': cosim,
                    'y_val': y_val
                })

        if not pairs_data:
            return

        title_suffix = f' (min interactions: {min_cooc_threshold})' if min_cooc_threshold > 0 else ''
        x_vals = np.array([p['co_occurrence'] for p in pairs_data])
        if use_softmax:
            base_vals = np.array([p['cosine_similarity'] for p in pairs_data])
            max_val = np.max(base_vals)
            exp_vals = np.exp(base_vals - max_val)
            y_vals = exp_vals / np.sum(exp_vals)
            for p, y in zip(pairs_data, y_vals):
                p['y_val'] = y
        else:
            y_vals = np.array([p['y_val'] for p in pairs_data])
        labels = [f"{p['char1']}-{p['char2']}" for p in pairs_data]

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_vals, y_vals, alpha=0.6, s=100, c=y_vals, cmap='viridis')

        # Labels like cooc_vs_cosine_scatter: try adjustText, else offset with bbox
        try:
            from adjustText import adjust_text
            texts = []
            for x, y, label in zip(x_vals, y_vals, labels):
                t = plt.annotate(label, (x, y), fontsize=7, alpha=0.8, ha='center', va='bottom')
                texts.append(t)
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
        except ImportError:
            x_range = x_vals.max() - x_vals.min() or 1
            y_range = y_vals.max() - y_vals.min() or 1
            offset_dist = min(x_range, y_range) * 0.08
            for idx, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
                angle = (idx * 137.5) % 360
                ox = offset_dist * np.cos(np.radians(angle))
                oy = offset_dist * np.sin(np.radians(angle))
                plt.annotate(
                    label, (x + ox, y + oy), fontsize=6, alpha=0.7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7, lw=0.5)
                )

        plt.xlabel('Interactions (co-occurrence count)', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f'{play_name} - Interactions vs {y_label}{title_suffix}', fontsize=14)
        plt.colorbar(scatter, label=y_label)
        plt.grid(True, alpha=0.3)
        if left_panel_text:
            plt.subplots_adjust(left=0.22)
            plt.gcf().text(
                0.04, 0.5,
                left_panel_text,
                fontsize=11, ha='left', va='center', rotation=0,
                transform=plt.gcf().transFigure,
            )
        if footer_text:
            plt.gcf().text(
                0.01, 0.01,
                footer_text,
                fontsize=9, ha='left', va='bottom'
            )
        if show_trend and len(x_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            plt.plot(x_vals, p(x_vals), "r--", alpha=0.8, label='Trend')
            plt.legend()
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        if output_path:
            outp = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(outp) or '.', exist_ok=True)
            plt.savefig(outp)
        else:
            os.makedirs(output_dir, exist_ok=True)
            path = output_dir.rstrip('/') + '/'
            plt.savefig(os.path.join(path, f'{safe_name}_{filename_suffix}.svg'))
        plt.close()

    def _plot_interactions_isolation_scatters(
        self,
        cosine_similarity_dict,
        play_name,
        ref_y_label,
        characters_filter=None,
        min_cooc_threshold=0,
        use_y_ratio=True,
        output_path_xy=None,
        output_path_dy=None,
    ):
        """
        For each character pair (same set as the normalized interactions scatter), X = co-occurrence.
        Y1 = mean Euclidean distance from this pair's point to every other pair's point in the
        original scatter plane (interactions × ref_y_value).
        Y2 = mean absolute difference in ref_y_value vs. every other pair (1D isolation in y).
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for isolation scatter plots")
        characters = list(self.co_occurrences.keys())
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        pairs_data = []
        for i, char1 in enumerate(characters):
            for char2 in characters[i + 1:]:
                cooc = self.co_occurrences[char1][char2]
                if cooc < min_cooc_threshold:
                    continue
                cosim = cosine_similarity_dict[char1][char2]
                if use_y_ratio and cooc == 0:
                    continue
                y_val = (cosim / cooc) if use_y_ratio else cosim
                pairs_data.append(
                    {
                        "char1": char1,
                        "char2": char2,
                        "co_occurrence": cooc,
                        "y_val": y_val,
                    }
                )

        if len(pairs_data) < 2:
            return

        x_vals = np.array([p["co_occurrence"] for p in pairs_data], dtype=float)
        y_vals = np.array([p["y_val"] for p in pairs_data], dtype=float)
        labels = [f"{p['char1']}-{p['char2']}" for p in pairs_data]

        xy = np.column_stack([x_vals, y_vals])
        diff = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.nan)
        mean_dist_xy = np.nanmean(dist, axis=1)

        dy = np.abs(y_vals[:, np.newaxis] - y_vals[np.newaxis, :])
        np.fill_diagonal(dy, np.nan)
        mean_dist_y = np.nanmean(dy, axis=1)

        title_suffix = (
            f" (min interactions: {min_cooc_threshold})" if min_cooc_threshold > 0 else ""
        )

        def _one_scatter(y_plot, y_label, out_path, footer):
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                x_vals, y_plot, alpha=0.6, s=100, c=y_plot, cmap="viridis"
            )
            try:
                from adjustText import adjust_text

                texts = []
                for x, y, lab in zip(x_vals, y_plot, labels):
                    t = plt.annotate(
                        lab, (x, y), fontsize=7, alpha=0.8, ha="center", va="bottom"
                    )
                    texts.append(t)
                adjust_text(
                    texts,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5, alpha=0.5),
                )
            except ImportError:
                x_range = x_vals.max() - x_vals.min() or 1
                y_range = y_plot.max() - y_plot.min() or 1
                offset_dist = min(x_range, y_range) * 0.08
                for idx, (x, y, lab) in enumerate(zip(x_vals, y_plot, labels)):
                    angle = (idx * 137.5) % 360
                    ox = offset_dist * np.cos(np.radians(angle))
                    oy = offset_dist * np.sin(np.radians(angle))
                    plt.annotate(
                        lab,
                        (x + ox, y + oy),
                        fontsize=6,
                        alpha=0.7,
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="gray",
                            alpha=0.7,
                            lw=0.5,
                        ),
                    )

            plt.xlabel("Interactions (co-occurrence count)", fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.title(
                f"{play_name} — {y_label}{title_suffix}", fontsize=14
            )
            plt.colorbar(scatter, label=y_label)
            plt.grid(True, alpha=0.3)
            plt.gcf().text(0.01, 0.01, footer, fontsize=9, ha="left", va="bottom")
            plt.tight_layout()
            outp = os.path.abspath(out_path)
            os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
            plt.savefig(outp)
            plt.close()

        if output_path_xy:
            _one_scatter(
                mean_dist_xy,
                f"Mean distance to other pairs (interactions × {ref_y_label})",
                output_path_xy,
                "Mean Euclidean distance from each point to all others in the interactions × "
                + ref_y_label
                + " plane.",
            )
        if output_path_dy:
            _one_scatter(
                mean_dist_y,
                f"Mean |Δ({ref_y_label})| vs. other pairs",
                output_path_dy,
                "Mean absolute difference in "
                + ref_y_label
                + " between this pair and every other pair.",
            )

    def _plot_similarity_scatter(
        self,
        x_similarity_dict,
        y_similarity_dict,
        play_name,
        x_label,
        y_label,
        filename_suffix,
        output_dir='output/',
        characters_filter=None,
        show_trend=True,
        show_labels=True,
        footer_text=None,
    ):
        """
        Internal helper: scatter plot with X = similarity per pair, Y = similarity per pair.
        """
        characters = list(self.characters)
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        pairs_data = []
        for i, char1 in enumerate(characters):
            for char2 in characters[i + 1:]:
                if char1 not in x_similarity_dict or char2 not in x_similarity_dict[char1]:
                    continue
                if char1 not in y_similarity_dict or char2 not in y_similarity_dict[char1]:
                    continue
                x_val = x_similarity_dict[char1][char2]
                y_val = y_similarity_dict[char1][char2]
                pairs_data.append({
                    'char1': char1, 'char2': char2,
                    'x_val': x_val, 'y_val': y_val
                })

        if not pairs_data:
            return

        x_vals = np.array([p['x_val'] for p in pairs_data], dtype=float)
        y_vals = np.array([p['y_val'] for p in pairs_data], dtype=float)
        labels = [f"{p['char1']}-{p['char2']}" for p in pairs_data]

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_vals, y_vals, alpha=0.65, s=100, c=y_vals, cmap='viridis')

        if show_labels:
            try:
                from adjustText import adjust_text
                texts = []
                for x, y, label in zip(x_vals, y_vals, labels):
                    t = plt.annotate(label, (x, y), fontsize=7, alpha=0.8, ha='center', va='bottom')
                    texts.append(t)
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
            except ImportError:
                x_range = x_vals.max() - x_vals.min() or 1
                y_range = y_vals.max() - y_vals.min() or 1
                offset_dist = min(x_range, y_range) * 0.08
                for idx, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
                    angle = (idx * 137.5) % 360
                    ox = offset_dist * np.cos(np.radians(angle))
                    oy = offset_dist * np.sin(np.radians(angle))
                    plt.annotate(
                        label, (x + ox, y + oy), fontsize=6, alpha=0.7, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7, lw=0.5)
                    )

        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f'{play_name} - {y_label} vs {x_label}', fontsize=14)
        plt.colorbar(scatter, label=y_label)
        plt.grid(True, alpha=0.3)
        if footer_text:
            plt.gcf().text(
                0.01, 0.01,
                footer_text,
                fontsize=9, ha='left', va='bottom'
            )
        if show_trend and len(x_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            plt.plot(x_vals, p(x_vals), "r--", alpha=0.8, label='Trend')
            plt.legend()
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{filename_suffix}.svg'))
        plt.close()

    def _plot_xy_scatter(
        self,
        x_vals,
        y_vals,
        play_name,
        x_label,
        y_label,
        filename_suffix,
        output_dir='output/',
        labels=None,
        show_labels=False,
        show_trend=True,
        x_tick_labels=None,
        footer_text=None,
    ):
        """
        Internal helper: scatter plot with X = x_vals, Y = y_vals.
        """
        if len(x_vals) == 0:
            return

        x_vals = np.array(x_vals, dtype=float)
        y_vals = np.array(y_vals, dtype=float)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_vals, y_vals, alpha=0.45, s=50, c=y_vals, cmap='viridis')

        if show_labels and labels:
            try:
                from adjustText import adjust_text
                texts = []
                for x, y, label in zip(x_vals, y_vals, labels):
                    t = plt.annotate(label, (x, y), fontsize=6, alpha=0.6, ha='center', va='bottom')
                    texts.append(t)
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
            except ImportError:
                pass

        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f'{play_name} - {y_label} vs {x_label}', fontsize=14)
        if x_tick_labels and len(x_tick_labels) == len(x_vals):
            plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation=45, ha='right', fontsize=8)
        plt.colorbar(scatter, label=y_label)
        plt.grid(True, alpha=0.3)
        if footer_text:
            plt.gcf().text(
                0.01, 0.01,
                footer_text,
                fontsize=9, ha='left', va='bottom'
            )
        if show_trend and len(x_vals) > 1:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            plt.plot(x_vals, p(x_vals), "r--", alpha=0.8, label='Trend')
            plt.legend()
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{filename_suffix}.svg'))
        plt.close()

    def __compute_interaction_cosine_mean(self, embedding_type='bert'):
        """
        Compute mean cosine similarity over consecutive speeches per character pair.
        """
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for interaction cosine normalization")

        interaction_similarity_sum = self.cosine_similarities[embedding_type]
        interaction_similarity_mean = {
            k: {k: 0 for k in self.characters} for k in self.characters
        }
        for c1 in self.characters:
            for c2 in self.characters:
                if c1 == c2:
                    continue
                cooc = self.co_occurrences.get(c1, {}).get(c2, 0)
                if cooc > 0:
                    interaction_similarity_mean[c1][c2] = interaction_similarity_sum[c1][c2] / cooc
                else:
                    interaction_similarity_mean[c1][c2] = 0.0
        return interaction_similarity_mean

    def __compute_scene_pair_stats(
        self,
        embedding_type='bert',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
    ):
        """
        Compute per-scene pair stats:
        - co_occurrence count within scene (consecutive speeches)
        - interaction cosine mean within scene
        - centroid cosine within scene
        Returns list of dicts with scene_idx, char1, char2, cooc, interaction_cos, centroid_cos.
        """
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        from scipy import spatial

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        if len(characters) == 0:
            return []

        embedding_key = f'average_{embedding_type}_embedding'
        results = []

        for scene_idx, scene in enumerate(self.characters_speeches):
            char_vectors = {char: [] for char in characters}
            char_counts = {char: 0 for char in characters}

            for speech in scene:
                speaker = speech.get('speaker', '[UNKNOWN]')
                if speaker not in char_vectors:
                    continue
                vec = speech.get(embedding_key)
                if vec is None:
                    continue
                try:
                    if hasattr(vec, 'detach'):
                        vec = vec.detach().cpu().numpy()
                except Exception:
                    pass
                char_vectors[speaker].append(np.array(vec))
                char_counts[speaker] += 1

            centroids = {}
            for char, vecs in char_vectors.items():
                if not vecs or char_counts[char] < min_speeches_per_scene:
                    centroids[char] = None
                else:
                    centroids[char] = np.mean(np.vstack(vecs), axis=0)

            pair_cos_sum = {c: {c2: 0.0 for c2 in characters} for c in characters}
            pair_counts = {c: {c2: 0 for c2 in characters} for c in characters}

            for speech_idx in range(len(scene) - 1):
                s1 = scene[speech_idx].get('speaker', '[UNKNOWN]')
                s2 = scene[speech_idx + 1].get('speaker', '[UNKNOWN]')
                if s1 == s2 or s1 not in characters or s2 not in characters:
                    continue
                v1 = scene[speech_idx].get(embedding_key)
                v2 = scene[speech_idx + 1].get(embedding_key)
                if v1 is None or v2 is None:
                    continue
                try:
                    if hasattr(v1, 'detach'):
                        v1 = v1.detach().cpu().numpy()
                    if hasattr(v2, 'detach'):
                        v2 = v2.detach().cpu().numpy()
                except Exception:
                    pass
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    cos_val = 1 - spatial.distance.cosine(v1, v2)
                else:
                    cos_val = 0.0
                pair_cos_sum[s1][s2] += cos_val
                pair_cos_sum[s2][s1] += cos_val
                pair_counts[s1][s2] += 1
                pair_counts[s2][s1] += 1

            for i, c1 in enumerate(characters):
                for c2 in characters[i + 1:]:
                    cooc = pair_counts[c1][c2]
                    if cooc == 0:
                        continue
                    v1, v2 = centroids.get(c1), centroids.get(c2)
                    if v1 is None or v2 is None:
                        continue
                    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        centroid_cos = 1 - spatial.distance.cosine(v1, v2)
                    else:
                        centroid_cos = 0.0
                    interaction_cos = pair_cos_sum[c1][c2] / cooc if cooc > 0 else 0.0
                    results.append({
                        'scene_idx': scene_idx,
                        'char1': c1,
                        'char2': c2,
                        'cooc': cooc,
                        'interaction_cos': interaction_cos,
                        'centroid_cos': centroid_cos,
                    })

        return results

    def plot_centroid_similarity_vs_interactions(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        min_cooc_threshold=0,
        show_trend=True,
    ):
        """
        Scatter plot:
        X = interactions (co-occurrence count),
        Y = cosine similarity between per-character centroid embeddings.
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for interactions scatter plot")
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        centroid_similarity = self._compute_centroid_similarity(embedding_type=embedding_type)
        self._plot_interactions_scatter(
            centroid_similarity,
            play_name,
            y_label=f'{embedding_type.upper()} centroid similarity',
            filename_suffix=f'{embedding_type}_centroid_similarity_vs_interactions',
            output_dir=output_dir,
            characters_filter=characters_filter,
            use_y_ratio=False,
            use_softmax=False,
            show_trend=show_trend,
            min_cooc_threshold=min_cooc_threshold,
            footer_text=rf'$\mathrm{{Centroid:}}\ \mathrm{{mean\ of\ all\ speeches}}$',
        )

    def plot_centroid_similarity_vs_interaction_cosine(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        show_trend=True,
    ):
        """
        Scatter plot:
        X = cosine similarity between consecutive speeches (mean over interactions),
        Y = cosine similarity between per-character centroid embeddings.
        """
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        centroid_similarity = self._compute_centroid_similarity(embedding_type=embedding_type)
        interaction_similarity_mean = self.__compute_interaction_cosine_mean(embedding_type=embedding_type)
        self._plot_similarity_scatter(
            interaction_similarity_mean,
            centroid_similarity,
            play_name,
            x_label=f'{embedding_type.upper()} interaction cosine (mean over consecutive speeches)',
            y_label=f'{embedding_type.upper()} centroid similarity',
            filename_suffix=f'{embedding_type}_centroid_vs_interaction_cosine_scatter',
            output_dir=output_dir,
            characters_filter=characters_filter,
            show_trend=show_trend,
            footer_text=(
                r'$\mathrm{X:}\ \mathrm{mean}\ \cos(\mathrm{consecutive\ speeches}),'
                r'\ \mathrm{Y:}\ \cos(\mu_i,\mu_j)$'
            ),
        )

    def plot_scene_centroid_similarity_vs_interaction_cosine(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
        show_trend=True,
        show_labels=False,
    ):
        """
        Scatter plot (per scene):
        X = mean cosine similarity between consecutive speeches per pair (global),
        Y = cosine similarity between per-scene character centroids.
        Each point corresponds to a character pair within a scene.
        """
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        from scipy import spatial

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) == 0:
            return

        interaction_similarity_mean = self.__compute_interaction_cosine_mean(embedding_type=embedding_type)
        embedding_key = f'average_{embedding_type}_embedding'

        x_vals = []
        y_vals = []
        labels = []
        for scene_idx, scene in enumerate(self.characters_speeches):
            char_vectors = {char: [] for char in characters}
            char_counts = {char: 0 for char in characters}
            for speech in scene:
                speaker = speech.get('speaker', '[UNKNOWN]')
                if speaker not in char_vectors:
                    continue
                vec = speech.get(embedding_key)
                if vec is None:
                    continue
                try:
                    if hasattr(vec, 'detach'):
                        vec = vec.detach().cpu().numpy()
                except Exception:
                    pass
                char_vectors[speaker].append(np.array(vec))
                char_counts[speaker] += 1

            centroids = {}
            for char, vecs in char_vectors.items():
                if not vecs or char_counts[char] < min_speeches_per_scene:
                    centroids[char] = None
                else:
                    centroids[char] = np.mean(np.vstack(vecs), axis=0)

            for i, c1 in enumerate(characters):
                for c2 in characters[i + 1:]:
                    v1, v2 = centroids.get(c1), centroids.get(c2)
                    if v1 is None or v2 is None:
                        continue
                    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        sim = 1 - spatial.distance.cosine(v1, v2)
                    else:
                        sim = 0.0
                    x_vals.append(interaction_similarity_mean[c1][c2])
                    y_vals.append(sim)
                    labels.append(f"{c1}-{c2}-S{scene_idx + 1}")

        self._plot_xy_scatter(
            x_vals,
            y_vals,
            play_name,
            x_label=f'{embedding_type.upper()} interaction cosine (mean over consecutive speeches)',
            y_label=f'{embedding_type.upper()} per-scene centroid similarity',
            filename_suffix=f'{embedding_type}_scene_centroid_vs_interaction_cosine_scatter',
            output_dir=output_dir,
            labels=labels,
            show_labels=show_labels,
            show_trend=show_trend,
            footer_text=(
                r'$\mathrm{X:}\ \mathrm{mean}\ \cos(\mathrm{consecutive\ speeches}),'
                r'\ \mathrm{Y:}\ \cos(\mu_i^{scene},\mu_j^{scene})$'
            ),
        )

    def plot_scene_interaction_cosine_over_scenes(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
        show_trend=True,
    ):
        """
        Line plot (per scene, per pair):
        X = scene index,
        Y = interaction cosine (mean within scene for each pair).
        """
        stats = self.__compute_scene_pair_stats(
            embedding_type=embedding_type,
            characters_filter=characters_filter,
            include_unknown=include_unknown,
            min_speeches_per_scene=min_speeches_per_scene,
        )
        if not stats:
            return

        scene_labels = self.__get_scene_labels()
        scene_count = len(self.characters_speeches)
        x_tick_labels = scene_labels if scene_labels and len(scene_labels) == scene_count else None

        pairs = sorted({(p['char1'], p['char2']) for p in stats})
        pair_to_series = {pair: [np.nan] * scene_count for pair in pairs}
        for p in stats:
            pair_to_series[(p['char1'], p['char2'])][p['scene_idx']] = p['interaction_cos']

        plt.figure(figsize=(12, 7))
        colors = plt.cm.get_cmap('tab20', max(1, len(pairs)))
        for idx, pair in enumerate(pairs):
            ys = pair_to_series[pair]
            xs = list(range(scene_count))
            plt.plot(xs, ys, marker='o', markersize=3, linewidth=1.2, color=colors(idx), label=f"{pair[0]}-{pair[1]}")

        if x_tick_labels:
            plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation=45, ha='right', fontsize=8)
        else:
            plt.xticks(range(scene_count))
        plt.xlabel('Scene index', fontsize=12)
        plt.ylabel(f'{embedding_type.upper()} interaction cosine (per pair)', fontsize=12)
        plt.title(f'{play_name} - Interaction Cosine Over Scenes ({embedding_type.upper()})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Y:}\ \mathrm{mean}\ \cos(\mathrm{consecutive\ speeches\ in\ scene})$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_scene_interaction_cosine_over_scenes.svg'))
        plt.close()

    def plot_scene_centroid_similarity_over_scenes(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
        show_trend=True,
    ):
        """
        Line plot (per scene, per pair):
        X = scene index,
        Y = centroid cosine similarity within scene (per pair).
        """
        stats = self.__compute_scene_pair_stats(
            embedding_type=embedding_type,
            characters_filter=characters_filter,
            include_unknown=include_unknown,
            min_speeches_per_scene=min_speeches_per_scene,
        )
        if not stats:
            return

        scene_labels = self.__get_scene_labels()
        scene_count = len(self.characters_speeches)
        x_tick_labels = scene_labels if scene_labels and len(scene_labels) == scene_count else None

        pairs = sorted({(p['char1'], p['char2']) for p in stats})
        pair_to_series = {pair: [np.nan] * scene_count for pair in pairs}
        for p in stats:
            pair_to_series[(p['char1'], p['char2'])][p['scene_idx']] = p['centroid_cos']

        plt.figure(figsize=(12, 7))
        colors = plt.cm.get_cmap('tab20', max(1, len(pairs)))
        for idx, pair in enumerate(pairs):
            ys = pair_to_series[pair]
            xs = list(range(scene_count))
            plt.plot(xs, ys, marker='o', markersize=3, linewidth=1.2, color=colors(idx), label=f"{pair[0]}-{pair[1]}")

        if x_tick_labels:
            plt.xticks(range(len(x_tick_labels)), x_tick_labels, rotation=45, ha='right', fontsize=8)
        else:
            plt.xticks(range(scene_count))
        plt.xlabel('Scene index', fontsize=12)
        plt.ylabel(f'{embedding_type.upper()} per-scene centroid similarity (per pair)', fontsize=12)
        plt.title(f'{play_name} - Centroid Similarity Over Scenes ({embedding_type.upper()})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Y:}\ \cos(\mu_i^{scene},\mu_j^{scene})$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_scene_centroid_similarity_over_scenes.svg'))
        plt.close()

    def plot_hidden_relationships_by_residual(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        min_cooc_threshold=0,
        top_n=20,
    ):
        """
        Rank pairs by positive residuals:
        residual = centroid_similarity - predicted_similarity(co_occurrence).
        Highlights pairs that are semantically close but interact less than expected.
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for hidden relationships plot")
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        centroid_similarity = self._compute_centroid_similarity(embedding_type=embedding_type)
        characters = list(self.co_occurrences.keys())
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        pairs = []
        for i, c1 in enumerate(characters):
            for c2 in characters[i + 1:]:
                cooc = self.co_occurrences[c1][c2]
                if cooc < min_cooc_threshold:
                    continue
                sim = centroid_similarity[c1][c2]
                pairs.append((c1, c2, cooc, sim))

        if not pairs:
            return []

        x = np.array([p[2] for p in pairs], dtype=float)
        y = np.array([p[3] for p in pairs], dtype=float)
        if len(x) > 1 and not np.allclose(x, x[0]):
            z = np.polyfit(x, y, 1)
            pred = z[0] * x + z[1]
        else:
            pred = np.full_like(y, y.mean() if len(y) else 0.0)
        residuals = y - pred

        ranked = sorted(
            [(p[0], p[1], p[2], p[3], r) for p, r in zip(pairs, residuals)],
            key=lambda v: v[4],
            reverse=True
        )
        if top_n is not None:
            ranked = ranked[: int(top_n)]

        labels = [f"{a}-{b}" for a, b, _, _, _ in ranked]
        values = [r for _, _, _, _, r in ranked]

        plt.figure(figsize=(max(12, len(labels) * 0.35), 8))
        bars = plt.bar(range(len(labels)), values, alpha=0.8)
        colors = plt.cm.RdYlGn((np.array(values) - min(values)) / (max(values) - min(values) + 1e-10))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Residual (centroid similarity - predicted)', fontsize=12)
        plt.xlabel('Character pairs', fontsize=12)
        plt.title(f'{play_name} - Hidden Relationships ({embedding_type.upper()} centroid residuals)', fontsize=14)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.grid(True, axis='y', alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Residual:}\ \cos(\mu_i,\mu_j)-\hat{\cos}(\mathrm{co\_occurrence})$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_hidden_relationships_residuals.svg'))
        plt.close()

        return [
            {
                'char1': a, 'char2': b,
                'co_occurrence': int(cooc),
                'centroid_similarity': float(sim),
                'residual': float(res),
            }
            for a, b, cooc, sim, res in ranked
        ]

    def plot_centroid_similarity_vs_graph_distance(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        min_cooc_threshold=1,
    ):
        """
        Scatter plot:
        X = shortest path length in interaction graph,
        Y = centroid cosine similarity.
        Highlights semantically close pairs that are far apart in the interaction graph.
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for graph-distance plot")
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        import networkx as nx

        centroid_similarity = self._compute_centroid_similarity(embedding_type=embedding_type)
        characters = list(self.co_occurrences.keys())
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        G = nx.Graph()
        G.add_nodes_from(characters)
        for i, c1 in enumerate(characters):
            for c2 in characters[i + 1:]:
                cooc = self.co_occurrences[c1][c2]
                if cooc >= min_cooc_threshold:
                    G.add_edge(c1, c2, weight=cooc)

        pairs = []
        for i, c1 in enumerate(characters):
            for c2 in characters[i + 1:]:
                if not nx.has_path(G, c1, c2):
                    continue
                dist = nx.shortest_path_length(G, c1, c2)
                sim = centroid_similarity[c1][c2]
                pairs.append((c1, c2, dist, sim))

        if not pairs:
            return []

        x_vals = np.array([p[2] for p in pairs], dtype=float)
        y_vals = np.array([p[3] for p in pairs], dtype=float)
        labels = [f"{p[0]}-{p[1]}" for p in pairs]

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_vals, y_vals, alpha=0.7, s=90, c=y_vals, cmap='viridis')
        try:
            from adjustText import adjust_text
            texts = []
            for x, y, label in zip(x_vals, y_vals, labels):
                t = plt.annotate(label, (x, y), fontsize=6, alpha=0.8, ha='center', va='bottom')
                texts.append(t)
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
        except ImportError:
            pass

        plt.xlabel('Shortest-path distance (interaction graph)', fontsize=12)
        plt.ylabel(f'{embedding_type.upper()} centroid similarity', fontsize=12)
        plt.title(f'{play_name} - Semantic Similarity vs Graph Distance', fontsize=14)
        plt.colorbar(scatter, label='Centroid similarity')
        plt.grid(True, alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Graph:}\ \mathrm{edges\ where\ co\_occurrence}\geq \tau$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_centroid_similarity_vs_graph_distance.svg'))
        plt.close()

        return [
            {
                'char1': a, 'char2': b,
                'graph_distance': int(dist),
                'centroid_similarity': float(sim),
            }
            for a, b, dist, sim in pairs
        ]

    def plot_semantic_outliers_zscore(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        min_cooc_threshold=0,
        top_n=20,
    ):
        """
        Bar plot of pairs with high semantic similarity but low interaction frequency.
        Score = z(similarity) - z(log1p(co_occurrence)).
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for semantic outliers plot")
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        centroid_similarity = self._compute_centroid_similarity(embedding_type=embedding_type)
        characters = list(self.co_occurrences.keys())
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        pairs = []
        for i, c1 in enumerate(characters):
            for c2 in characters[i + 1:]:
                cooc = self.co_occurrences[c1][c2]
                if cooc < min_cooc_threshold:
                    continue
                sim = centroid_similarity[c1][c2]
                pairs.append((c1, c2, cooc, sim))

        if not pairs:
            return []

        sims = np.array([p[3] for p in pairs], dtype=float)
        coocs = np.array([p[2] for p in pairs], dtype=float)
        log_coocs = np.log1p(coocs)

        sim_mean, sim_std = sims.mean(), sims.std()
        cooc_mean, cooc_std = log_coocs.mean(), log_coocs.std()
        z_sim = (sims - sim_mean) / (sim_std if sim_std > 0 else 1.0)
        z_cooc = (log_coocs - cooc_mean) / (cooc_std if cooc_std > 0 else 1.0)
        scores = z_sim - z_cooc

        ranked = sorted(
            [(p[0], p[1], p[2], p[3], s) for p, s in zip(pairs, scores)],
            key=lambda v: v[4],
            reverse=True
        )
        if top_n is not None:
            ranked = ranked[: int(top_n)]

        labels = [f"{a}-{b}" for a, b, _, _, _ in ranked]
        values = [s for _, _, _, _, s in ranked]

        plt.figure(figsize=(max(12, len(labels) * 0.35), 8))
        bars = plt.bar(range(len(labels)), values, alpha=0.85)
        colors = plt.cm.coolwarm((np.array(values) - min(values)) / (max(values) - min(values) + 1e-10))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Outlier score (z_sim - z_log_cooc)', fontsize=12)
        plt.xlabel('Character pairs', fontsize=12)
        plt.title(f'{play_name} - Semantic Outliers ({embedding_type.upper()} centroid)', fontsize=14)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.grid(True, axis='y', alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Score:}\ z(\cos(\mu_i,\mu_j)) - z(\log(1+\mathrm{co\_occurrence}))$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_semantic_outliers_zscore.svg'))
        plt.close()

        return [
            {
                'char1': a, 'char2': b,
                'co_occurrence': int(cooc),
                'centroid_similarity': float(sim),
                'outlier_score': float(score),
            }
            for a, b, cooc, sim, score in ranked
        ]

    def plot_scene_drift_heatmap(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
    ):
        """
        Heatmap of per-character semantic drift across consecutive scenes.
        Drift is 1 - cosine similarity between a character's scene centroids.
        """
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        from scipy import spatial
        import seaborn as sns

        scene_centroids = self.__compute_scene_character_centroids(embedding_type=embedding_type)
        scene_labels = self.__get_scene_labels()
        if scene_labels and len(scene_labels) != len(scene_centroids):
            scene_labels = []

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) == 0 or len(scene_centroids) < 2:
            return []

        drift_matrix = []
        for char in characters:
            row = []
            prev_vec = None
            prev_count = 0
            for scene_idx, scene in enumerate(self.characters_speeches):
                # Count speeches for min threshold
                count = sum(1 for sp in scene if sp.get('speaker') == char)
                vec = scene_centroids[scene_idx].get(char)
                if vec is None or count < min_speeches_per_scene:
                    row.append(np.nan)
                    prev_vec = None
                    prev_count = 0
                    continue

                if prev_vec is None or prev_count < min_speeches_per_scene:
                    row.append(np.nan)
                else:
                    norm1, norm2 = np.linalg.norm(prev_vec), np.linalg.norm(vec)
                    if norm1 > 0 and norm2 > 0:
                        sim = 1 - spatial.distance.cosine(prev_vec, vec)
                        row.append(1 - sim)
                    else:
                        row.append(np.nan)
                prev_vec = vec
                prev_count = count
            drift_matrix.append(row)

        drift_array = np.array(drift_matrix, dtype=float)
        plt.figure(figsize=(max(12, len(scene_centroids) * 0.35), max(6, len(characters) * 0.35)))
        sns.heatmap(
            drift_array,
            cmap='magma',
            mask=np.isnan(drift_array),
            cbar_kws={'label': 'Semantic drift (1 - cosine)'},
            xticklabels=scene_labels if scene_labels else [f"S{i+1}" for i in range(len(scene_centroids))],
            yticklabels=characters,
        )
        plt.xlabel('Scene index', fontsize=12)
        plt.ylabel('Character', fontsize=12)
        plt.title(f'{play_name} - Scene-by-Scene Semantic Drift ({embedding_type.upper()})', fontsize=14)
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_scene_drift_heatmap.svg'))
        plt.close()

        results = []
        for i, char in enumerate(characters):
            for j in range(len(scene_centroids)):
                if np.isnan(drift_array[i, j]):
                    continue
                results.append({
                    'character': char,
                    'scene_index': j,
                    'scene_label': scene_labels[j] if scene_labels else f"S{j+1}",
                    'drift': float(drift_array[i, j]),
                })
        return results

    def plot_scene_embedding_trajectories(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
    ):
        """
        PCA trajectory plot of per-scene character centroids.
        Each character is a line through scene-wise PCA positions.
        """
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        scene_centroids = self.__compute_scene_character_centroids(embedding_type=embedding_type)
        scene_labels = self.__get_scene_labels()
        if scene_labels and len(scene_labels) != len(scene_centroids):
            scene_labels = []

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) == 0 or len(scene_centroids) < 2:
            return []

        points = []
        meta = []
        for char in characters:
            for scene_idx, scene in enumerate(self.characters_speeches):
                count = sum(1 for sp in scene if sp.get('speaker') == char)
                if count < min_speeches_per_scene:
                    continue
                vec = scene_centroids[scene_idx].get(char)
                if vec is None:
                    continue
                points.append(vec)
                meta.append((char, scene_idx))

        if len(points) < 2:
            return []

        point_matrix = np.vstack(points)
        if len(points) < 2 or np.allclose(point_matrix, point_matrix[0]):
            pcs = np.zeros((len(points), 2))
        else:
            pcs = PCA(n_components=2).fit_transform(point_matrix)

        char_to_points = {c: [] for c in characters}
        for (char, scene_idx), (x, y) in zip(meta, pcs):
            char_to_points[char].append((scene_idx, x, y))

        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('tab10', len(characters))
        for idx, char in enumerate(characters):
            seq = sorted(char_to_points[char], key=lambda v: v[0])
            if len(seq) < 2:
                continue
            xs = [p[1] for p in seq]
            ys = [p[2] for p in seq]
            plt.plot(xs, ys, marker='o', markersize=4, linewidth=1.5, color=colors(idx), label=char)

        plt.xlabel('PCA dimension 1 (scene centroids)', fontsize=12)
        plt.ylabel('PCA dimension 2 (scene centroids)', fontsize=12)
        plt.title(f'{play_name} - Scene Trajectories ({embedding_type.upper()})', fontsize=14)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_scene_trajectories.svg'))
        plt.close()

        return [
            {
                'character': char,
                'scene_index': int(scene_idx),
                'scene_label': scene_labels[scene_idx] if scene_labels else f"S{scene_idx+1}",
                'pc1': float(x),
                'pc2': float(y),
            }
            for (char, scene_idx), (x, y) in zip(meta, pcs)
        ]

    def plot_bert_interactions_scatter(self, play_name, output_dir='output/', characters_filter=None, min_cooc_threshold=0):
        """
        Scatter plot: X = interactions (co-occurrence count) per pair,
        Y = BERT cosine similarity sum (raw). No trend line.

        Args:
            min_cooc_threshold: Minimum interactions to include a pair (default: 0).
        """
        if "bert" not in self.cosine_similarities:
            raise ValueError("bert option required for BERT scatter plot")
        self._plot_interactions_scatter(
            self.cosine_similarities["bert"],
            play_name,
            y_label='BERT Similarity Sum',
            filename_suffix='bert_interactions_scatter',
            output_dir=output_dir,
            characters_filter=characters_filter,
            use_y_ratio=False,
            use_softmax=False,
            show_trend=False,
            min_cooc_threshold=min_cooc_threshold,
        )

    def plot_bert_interactions_scatter_normalized(self, play_name, output_dir='output/', characters_filter=None, min_cooc_threshold=0):
        """
        Scatter plot: X = interactions (co-occurrence count) per pair,
        Y = BERT Cosine Similarity / Interactions (normalized). No trend line.

        Args:
            min_cooc_threshold: Minimum interactions to include a pair (default: 0).
        """
        if "bert" not in self.cosine_similarities:
            raise ValueError("bert option required for BERT scatter plot")
        self._plot_interactions_scatter(
            self.cosine_similarities["bert"],
            play_name,
            y_label='BERT Cosine Similarity / Interactions',
            filename_suffix='bert_interactions_scatter_normalized',
            output_dir=output_dir,
            characters_filter=characters_filter,
            use_y_ratio=True,
            use_softmax=False,
            show_trend=False,
            min_cooc_threshold=min_cooc_threshold,
        )

    def plot_w2v_interactions_scatter(self, play_name, output_dir='output/', characters_filter=None, min_cooc_threshold=0):
        """
        Scatter plot: X = interactions (co-occurrence count) per pair,
        Y = W2V cosine similarity between mean-centered character NAME embeddings, softmax-normalized across pairs.
        No trend line. Not a sum over interactions—just cosine between name vectors.

        Args:
            min_cooc_threshold: Minimum interactions to include a pair (default: 0).
        """
        if "w2v" not in self.options:
            raise ValueError("w2v option required for W2V scatter plot")
        w2v_params = getattr(self, '_w2v_params', {'vector_size': 100, 'window_size': 10})
        self._plot_interactions_scatter(
            self.cosine_similarity_w2v,
            play_name,
            y_label='W2V Similarity (mean-centered)',
            filename_suffix='w2v_interactions_scatter',
            output_dir=output_dir,
            characters_filter=characters_filter,
            use_y_ratio=False,
            use_softmax=False,
            show_trend=False,
            min_cooc_threshold=min_cooc_threshold,
            footer_text=rf'$\mathrm{{W2V:}}\ \mathrm{{window}}={w2v_params["window_size"]},\ \mathrm{{dim}}={w2v_params["vector_size"]}$',
        )

    def plot_scene_pca1_over_time(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
    ):
        """
        Line plot: X = scene index, Y = PCA1 of per-scene character centroids.
        Each character has a line across scenes where they speak.
        """
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        scene_centroids = self.__compute_scene_character_centroids(embedding_type=embedding_type)
        scene_labels = self.__get_scene_labels()
        if scene_labels and len(scene_labels) != len(scene_centroids):
            scene_labels = []

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) == 0 or len(scene_centroids) < 2:
            return []

        points = []
        meta = []
        for char in characters:
            for scene_idx, scene in enumerate(self.characters_speeches):
                count = sum(1 for sp in scene if sp.get('speaker') == char)
                if count < min_speeches_per_scene:
                    continue
                vec = scene_centroids[scene_idx].get(char)
                if vec is None:
                    continue
                points.append(vec)
                meta.append((char, scene_idx))

        if len(points) < 2:
            return []

        point_matrix = np.vstack(points)
        if np.allclose(point_matrix, point_matrix[0]):
            pc1 = np.zeros(len(points))
        else:
            pc1 = PCA(n_components=1).fit_transform(point_matrix).ravel()

        char_to_series = {c: {} for c in characters}
        for (char, scene_idx), val in zip(meta, pc1):
            char_to_series[char][scene_idx] = val

        plt.figure(figsize=(12, 7))
        colors = plt.cm.get_cmap('tab10', len(characters))
        for idx, char in enumerate(characters):
            if len(char_to_series[char]) < 2:
                continue
            xs = list(range(len(scene_centroids)))
            ys = [np.nan] * len(scene_centroids)
            for scene_idx, val in char_to_series[char].items():
                ys[scene_idx] = val
            # NaNs break the line, so we don't connect across absent scenes.
            plt.plot(xs, ys, marker='o', markersize=4, linewidth=1.6, color=colors(idx), label=char)

        if scene_labels:
            plt.xticks(range(len(scene_labels)), scene_labels, rotation=45, ha='right', fontsize=8)
        else:
            plt.xticks(range(len(scene_centroids)))

        plt.xlabel('Scene index', fontsize=12)
        plt.ylabel('PCA1 (scene centroids)', fontsize=12)
        plt.title(f'{play_name} - PCA1 Over Scenes ({embedding_type.upper()})', fontsize=14)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_scene_pca1_over_time.svg'))
        plt.close()

        return [
            {
                'character': char,
                'scene_index': int(scene_idx),
                'scene_label': scene_labels[scene_idx] if scene_labels else f"S{scene_idx+1}",
                'pc1': float(val),
            }
            for (char, scene_idx), val in zip(meta, pc1)
        ]

    def plot_copresence_hidden_pairs(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
        min_copresence=2,
        top_n=15,
    ):
        """
        Bar chart of pairs that share scenes but rarely interact.
        Score favors high centroid similarity and high co-presence, penalizes direct interactions.
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for co-presence plot")
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        copresence = self.__calculate_scene_copresence(
            min_speeches_per_scene=min_speeches_per_scene,
            include_unknown=include_unknown,
        )
        centroid_similarity = self._compute_centroid_similarity(embedding_type=embedding_type)

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) < 2:
            return []

        pairs = []
        for i, c1 in enumerate(characters):
            for c2 in characters[i + 1:]:
                cooc = self.co_occurrences[c1][c2]
                co_scene = copresence[c1][c2]
                if co_scene < min_copresence:
                    continue
                sim = centroid_similarity[c1][c2]
                score = sim * np.log1p(co_scene) / (1.0 + cooc)
                pairs.append({
                    'char1': c1,
                    'char2': c2,
                    'co_presence': int(co_scene),
                    'co_occurrence': int(cooc),
                    'centroid_similarity': float(sim),
                    'score': float(score),
                })

        if not pairs:
            return []

        pairs = sorted(pairs, key=lambda r: r['score'], reverse=True)
        if top_n is not None:
            pairs = pairs[: int(top_n)]

        labels = [f"{p['char1']}-{p['char2']}" for p in pairs]
        values = [p['score'] for p in pairs]

        plt.figure(figsize=(max(12, len(labels) * 0.4), 6))
        bars = plt.bar(range(len(labels)), values, alpha=0.85)
        colors = plt.cm.viridis(
            (np.array(values) - min(values)) / (max(values) - min(values) + 1e-10)
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Hidden co-presence score', fontsize=12)
        plt.xlabel('Character pairs', fontsize=12)
        plt.title(f'{play_name} - Co-presence Hidden Pairs ({embedding_type.upper()})', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Score:}\ \cos(\mu_i,\mu_j)\cdot\log(1+\mathrm{co\_scene})/(1+\mathrm{co\_occ})$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_copresence_hidden_pairs.svg'))
        plt.close()

        return pairs

    def plot_lagged_scene_similarity_pairs(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_speeches_per_scene=1,
        min_lag_count=2,
        top_n=15,
    ):
        """
        Bar chart of pairs with high semantic similarity across adjacent scenes.
        Measures cross-scene "echo" (scene i for char A vs scene i+1 for char B).
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for lagged similarity plot")
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        from scipy import spatial

        scene_centroids = self.__compute_scene_character_centroids(embedding_type=embedding_type)
        if len(scene_centroids) < 2:
            return []

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) < 2:
            return []

        pair_scores = {}
        pair_counts = {}

        for i in range(len(scene_centroids) - 1):
            scene_a = self.characters_speeches[i]
            scene_b = self.characters_speeches[i + 1]
            present_a = {
                c for c in characters
                if sum(1 for sp in scene_a if sp.get('speaker') == c) >= min_speeches_per_scene
            }
            present_b = {
                c for c in characters
                if sum(1 for sp in scene_b if sp.get('speaker') == c) >= min_speeches_per_scene
            }
            for c1 in present_a:
                v1 = scene_centroids[i].get(c1)
                if v1 is None:
                    continue
                for c2 in present_b:
                    v2 = scene_centroids[i + 1].get(c2)
                    if v2 is None:
                        continue
                    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if norm1 <= 0 or norm2 <= 0:
                        continue
                    sim = 1 - spatial.distance.cosine(v1, v2)
                    key = tuple(sorted((c1, c2)))
                    pair_scores[key] = pair_scores.get(key, 0.0) + sim
                    pair_counts[key] = pair_counts.get(key, 0) + 1

        pairs = []
        for (c1, c2), total_sim in pair_scores.items():
            count = pair_counts[(c1, c2)]
            if count < min_lag_count:
                continue
            mean_sim = total_sim / count
            cooc = self.co_occurrences[c1][c2]
            score = mean_sim / (1.0 + cooc)
            pairs.append({
                'char1': c1,
                'char2': c2,
                'lag_count': int(count),
                'lag_similarity': float(mean_sim),
                'co_occurrence': int(cooc),
                'score': float(score),
            })

        if not pairs:
            return []

        pairs = sorted(pairs, key=lambda r: r['score'], reverse=True)
        if top_n is not None:
            pairs = pairs[: int(top_n)]

        labels = [f"{p['char1']}-{p['char2']}" for p in pairs]
        values = [p['score'] for p in pairs]

        plt.figure(figsize=(max(12, len(labels) * 0.4), 6))
        bars = plt.bar(range(len(labels)), values, alpha=0.85)
        colors = plt.cm.plasma(
            (np.array(values) - min(values)) / (max(values) - min(values) + 1e-10)
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Lagged similarity score', fontsize=12)
        plt.xlabel('Character pairs', fontsize=12)
        plt.title(f'{play_name} - Cross-Scene Echo Pairs ({embedding_type.upper()})', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Score:}\ \mathrm{mean\_lag\_sim}/(1+\mathrm{co\_occ})$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_lagged_echo_pairs.svg'))
        plt.close()

        return pairs

    def plot_role_similarity_hidden_pairs(
        self,
        play_name,
        embedding_type='bert',
        output_dir='output/',
        characters_filter=None,
        include_unknown=False,
        min_total_speeches=5,
        alpha=0.5,
        top_n=20,
    ):
        """
        Bar chart of pairs with similar "roles" in the play:
        - Interaction-profile similarity: cosine between co-occurrence vectors.
        - Semantic-profile similarity: cosine between centroid-similarity vectors.
        Combined score = alpha * interaction_sim + (1-alpha) * semantic_sim.

        Hidden connections are pairs with high combined role similarity regardless
        of direct interaction count.
        """
        if "co-oc" not in self.options:
            raise ValueError("co-oc option required for role similarity plot")
        if embedding_type not in self.cosine_similarities:
            raise ValueError(
                f"embedding_type '{embedding_type}' not computed. "
                f"Available: {list(self.cosine_similarities)}")

        from scipy import spatial

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        speech_counts = {char: 0 for char in characters}
        for scene in self.characters_speeches:
            for speech in scene:
                speaker = speech.get('speaker', '[UNKNOWN]')
                if speaker in speech_counts:
                    speech_counts[speaker] += 1

        characters = [c for c in characters if speech_counts.get(c, 0) >= min_total_speeches]
        if len(characters) < 2:
            return []

        centroid_similarity = self._compute_centroid_similarity(embedding_type=embedding_type)

        interaction_profiles = {}
        semantic_profiles = {}
        for char in characters:
            inter_vec = []
            sem_vec = []
            for other in characters:
                if other == char:
                    continue
                inter_vec.append(np.log1p(self.co_occurrences[char][other]))
                sem_vec.append(centroid_similarity[char][other])
            interaction_profiles[char] = np.array(inter_vec, dtype=float)
            semantic_profiles[char] = np.array(sem_vec, dtype=float)

        pairs = []
        for i, c1 in enumerate(characters):
            for c2 in characters[i + 1:]:
                v1 = interaction_profiles[c1]
                v2 = interaction_profiles[c2]
                s1 = semantic_profiles[c1]
                s2 = semantic_profiles[c2]

                inter_sim = 0.0
                sem_sim = 0.0
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    inter_sim = 1 - spatial.distance.cosine(v1, v2)
                if np.linalg.norm(s1) > 0 and np.linalg.norm(s2) > 0:
                    sem_sim = 1 - spatial.distance.cosine(s1, s2)

                score = alpha * inter_sim + (1.0 - alpha) * sem_sim
                cooc = self.co_occurrences[c1][c2]
                pairs.append({
                    'char1': c1,
                    'char2': c2,
                    'interaction_profile_sim': float(inter_sim),
                    'semantic_profile_sim': float(sem_sim),
                    'score': float(score),
                    'co_occurrence': int(cooc),
                })

        if not pairs:
            return []

        pairs = sorted(pairs, key=lambda r: r['score'], reverse=True)
        if top_n is not None:
            pairs = pairs[: int(top_n)]

        labels = [f"{p['char1']}-{p['char2']}" for p in pairs]
        values = [p['score'] for p in pairs]
        cooc_vals = np.array([p['co_occurrence'] for p in pairs], dtype=float)
        color_vals = np.log1p(cooc_vals)

        plt.figure(figsize=(max(12, len(labels) * 0.4), 6))
        bars = plt.bar(range(len(labels)), values, alpha=0.85)
        colors = plt.cm.viridis(
            (color_vals - color_vals.min()) / (color_vals.max() - color_vals.min() + 1e-10)
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Role similarity score', fontsize=12)
        plt.xlabel('Character pairs', fontsize=12)
        plt.title(f'{play_name} - Role Similarity Hidden Pairs ({embedding_type.upper()})', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Score:}\ \alpha\cdot\cos(\mathrm{co\_occ\_profiles})+(1-\alpha)\cdot\cos(\mathrm{semantic\_profiles})$',
            fontsize=9, ha='left', va='bottom'
        )
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap='viridis')
        sm.set_array(color_vals)
        plt.colorbar(sm, ax=plt.gca(), label='Direct co-occurrence (log1p) intensity')
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_{embedding_type}_role_similarity_hidden_pairs.svg'))
        plt.close()

        return pairs

    def plot_name_pca_vs_speeches(
        self,
        play_name,
        output_dir='output/',
        characters_filter=None,
        raw_text_dir='Data/raw_text',
        vector_size=100,
        window_size=10,
        include_unknown=False,
    ):
        """
        Scatter plot per character:
        X = PCA dimension 1 of character-name vectors (from W2V on raw text),
        Y = number of speeches by that character.
        """
        from W2V import TrainW2VModel

        raw_text_path = self.__get_raw_text_path(raw_text_dir)
        if not raw_text_path or not os.path.exists(raw_text_path):
            code = os.path.splitext(os.path.basename(self.xml_path))[0]
            raise FileNotFoundError(
                f"Raw text from Folger not found for {code}. "
                f"Run: conda activate mind-the-gap && python src/download_folger_raw_text.py {code}"
            )

        corpus_list = self.__load_raw_text_tokens(raw_text_path)
        if len(corpus_list) < 10:
            return []

        book_name = os.path.splitext(os.path.basename(self.xml_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir if output_dir.endswith('/') else output_dir + '/'
        w2v_vectors = TrainW2VModel(
            book_name, corpus_list, vector_size, window_size, output_path, forward_only=False
        )
        vectors = w2v_vectors.get("vectors", {})

        def char_id_to_token(char_id):
            if char_id == '[UNKNOWN]':
                return 'unknown'
            part = char_id.split('_')[0].split('.')[-1]
            return part.lower()

        if vectors:
            target_dim = max(len(v) for v in vectors.values() if hasattr(v, '__len__'))
        else:
            target_dim = vector_size
        speech_counts = {char: 0 for char in self.characters}
        for scene in self.characters_speeches:
            for speech in scene:
                speaker = speech.get('speaker', '[UNKNOWN]')
                if speaker in speech_counts:
                    speech_counts[speaker] += 1

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]

        if not characters:
            return []

        name_vectors = []
        counts = []
        labels = []
        for char in characters:
            token = char_id_to_token(char)
            if token in vectors:
                vec = np.array(vectors[token])
            else:
                vec = np.zeros(target_dim)
            if vec.size != target_dim:
                if vec.size == 0:
                    vec = np.zeros(target_dim)
                elif vec.size > target_dim:
                    vec = vec[:target_dim]
                else:
                    vec = np.pad(vec, (0, target_dim - vec.size), mode='constant')
            name_vectors.append(vec)
            counts.append(speech_counts.get(char, 0))
            labels.append(char)

        name_matrix = np.vstack(name_vectors)
        if len(name_vectors) < 2 or np.allclose(name_matrix, name_matrix[0]):
            pc1 = np.zeros(len(name_vectors))
        else:
            pc1 = PCA(n_components=1).fit_transform(name_matrix).ravel()

        x_vals = np.array(pc1)
        y_vals = np.array(counts)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_vals, y_vals, alpha=0.7, s=110, c=y_vals, cmap='viridis')
        try:
            from adjustText import adjust_text
            texts = []
            for x, y, label in zip(x_vals, y_vals, labels):
                t = plt.annotate(label, (x, y), fontsize=7, alpha=0.85, ha='center', va='bottom')
                texts.append(t)
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
        except ImportError:
            x_range = x_vals.max() - x_vals.min() or 1
            y_range = y_vals.max() - y_vals.min() or 1
            offset_dist = min(x_range, y_range) * 0.08
            for idx, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
                angle = (idx * 137.5) % 360
                ox = offset_dist * np.cos(np.radians(angle))
                oy = offset_dist * np.sin(np.radians(angle))
                plt.annotate(
                    label, (x + ox, y + oy), fontsize=6, alpha=0.7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7, lw=0.5)
                )

        plt.xlabel('PCA dimension 1 (character-name vectors)', fontsize=12)
        plt.ylabel('Number of speeches', fontsize=12)
        plt.title(f'{play_name} - Character Name PCA vs Speech Count', fontsize=14)
        plt.colorbar(scatter, label='Number of speeches')
        plt.grid(True, alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Formula:}\ x=\mathrm{PCA}_1(\mathrm{name\ vectors}),\ y=\#\mathrm{speeches}$'
            + '\n'
            + rf'$\mathrm{{W2V:}}\ \mathrm{{window}}={window_size},\ \mathrm{{dim}}={vector_size}$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_name_pca_vs_speeches.svg'))
        plt.close()

        return [{'character': c, 'pca_dim1': float(x), 'speech_count': int(y)}
                for c, x, y in zip(labels, x_vals, y_vals)]

    def plot_name_w2v_difference_pca_scatter(
        self,
        play_name,
        output_dir='output/',
        characters_filter=None,
        raw_text_dir='Data/raw_text',
        vector_size=100,
        window_size=10,
        include_unknown=False,
    ):
        """
        Scatter plot of pairwise differences between character-name vectors (W2V):
        1) Train W2V on raw text
        2) Build name vectors for characters
        3) Compute difference vectors (v_i - v_j) for all pairs
        4) Plot PCA PC1 vs PC2 of those differences
        """
        from W2V import TrainW2VModel

        raw_text_path = self.__get_raw_text_path(raw_text_dir)
        if not raw_text_path or not os.path.exists(raw_text_path):
            code = os.path.splitext(os.path.basename(self.xml_path))[0]
            raise FileNotFoundError(
                f"Raw text from Folger not found for {code}. "
                f"Run: conda activate mind-the-gap && python src/download_folger_raw_text.py {code}"
            )

        corpus_list = self.__load_raw_text_tokens(raw_text_path)
        if len(corpus_list) < 10:
            return []

        book_name = os.path.splitext(os.path.basename(self.xml_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir if output_dir.endswith('/') else output_dir + '/'
        w2v_vectors = TrainW2VModel(
            book_name, corpus_list, vector_size, window_size, output_path, forward_only=False
        )
        vectors = w2v_vectors.get("vectors", {})

        def char_id_to_token(char_id):
            if char_id == '[UNKNOWN]':
                return 'unknown'
            part = char_id.split('_')[0].split('.')[-1]
            return part.lower()

        if vectors:
            target_dim = max(len(v) for v in vectors.values() if hasattr(v, '__len__'))
        else:
            target_dim = vector_size

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) < 2:
            return []

        name_vectors = {}
        for char in characters:
            token = char_id_to_token(char)
            if token in vectors:
                vec = np.array(vectors[token])
            else:
                vec = np.zeros(target_dim)
            if vec.size != target_dim:
                if vec.size == 0:
                    vec = np.zeros(target_dim)
                elif vec.size > target_dim:
                    vec = vec[:target_dim]
                else:
                    vec = np.pad(vec, (0, target_dim - vec.size), mode='constant')
            name_vectors[char] = vec

        pair_labels = []
        diff_vectors = []
        for i, left in enumerate(characters):
            for right in characters[i + 1:]:
                diff_vectors.append(name_vectors[left] - name_vectors[right])
                pair_labels.append(f"{left}-{right}")

        if not diff_vectors:
            return []

        diff_matrix = np.vstack(diff_vectors)
        if len(diff_vectors) < 2 or np.allclose(diff_matrix, diff_matrix[0]):
            pcs = np.zeros((len(diff_vectors), 2))
        else:
            pcs = PCA(n_components=2).fit_transform(diff_matrix)

        x_vals = pcs[:, 0]
        y_vals = pcs[:, 1]
        norms = np.linalg.norm(diff_matrix, axis=1)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_vals, y_vals, alpha=0.75, s=90, c=norms, cmap='plasma')
        try:
            from adjustText import adjust_text
            texts = []
            for x, y, label in zip(x_vals, y_vals, pair_labels):
                t = plt.annotate(label, (x, y), fontsize=6, alpha=0.85, ha='center', va='bottom')
                texts.append(t)
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
        except ImportError:
            x_range = x_vals.max() - x_vals.min() or 1
            y_range = y_vals.max() - y_vals.min() or 1
            offset_dist = min(x_range, y_range) * 0.05
            for idx, (x, y, label) in enumerate(zip(x_vals, y_vals, pair_labels)):
                angle = (idx * 137.5) % 360
                ox = offset_dist * np.cos(np.radians(angle))
                oy = offset_dist * np.sin(np.radians(angle))
                plt.annotate(
                    label, (x + ox, y + oy), fontsize=5, alpha=0.7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.7, lw=0.4)
                )

        plt.xlabel('PCA dimension 1 (name-vector differences)', fontsize=12)
        plt.ylabel('PCA dimension 2 (name-vector differences)', fontsize=12)
        plt.title(f'{play_name} - W2V Name-Vector Difference PCA', fontsize=14)
        plt.colorbar(scatter, label='Difference vector L2 norm')
        plt.grid(True, alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Formula:}\ \Delta v = v_i - v_j,\ \mathrm{plot}\ \mathrm{PCA}_{1,2}(\Delta v)$'
            + '\n'
            + rf'$\mathrm{{W2V:}}\ \mathrm{{window}}={window_size},\ \mathrm{{dim}}={vector_size}$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_name_w2v_diff_pca_scatter.svg'))
        plt.close()

        return [
            {'pair': label, 'pc1': float(x), 'pc2': float(y), 'diff_norm': float(n)}
            for label, x, y, n in zip(pair_labels, x_vals, y_vals, norms)
        ]

    def plot_name_w2v_mean_centered_pca_scatter(
        self,
        play_name,
        output_dir='output/',
        characters_filter=None,
        raw_text_dir='Data/raw_text',
        vector_size=100,
        window_size=10,
        include_unknown=False,
    ):
        """
        Scatter plot of mean-centered character-name vectors (W2V):
        1) Train W2V on raw text
        2) Build name vectors for characters
        3) Subtract mean(name vectors) from each name vector
        4) Plot PCA PC1 vs PC2 of those centered vectors
        """
        from W2V import TrainW2VModel

        raw_text_path = self.__get_raw_text_path(raw_text_dir)
        if not raw_text_path or not os.path.exists(raw_text_path):
            code = os.path.splitext(os.path.basename(self.xml_path))[0]
            raise FileNotFoundError(
                f"Raw text from Folger not found for {code}. "
                f"Run: conda activate mind-the-gap && python src/download_folger_raw_text.py {code}"
            )

        corpus_list = self.__load_raw_text_tokens(raw_text_path)
        if len(corpus_list) < 10:
            return []

        book_name = os.path.splitext(os.path.basename(self.xml_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir if output_dir.endswith('/') else output_dir + '/'
        w2v_vectors = TrainW2VModel(
            book_name, corpus_list, vector_size, window_size, output_path, forward_only=False
        )
        vectors = w2v_vectors.get("vectors", {})

        def char_id_to_token(char_id):
            if char_id == '[UNKNOWN]':
                return 'unknown'
            part = char_id.split('_')[0].split('.')[-1]
            return part.lower()

        if vectors:
            target_dim = max(len(v) for v in vectors.values() if hasattr(v, '__len__'))
        else:
            target_dim = vector_size

        characters = list(self.characters)
        if not include_unknown:
            characters = [c for c in characters if c != '[UNKNOWN]']
        if characters_filter is not None:
            characters = [c for c in characters if c in characters_filter]
        if len(characters) < 2:
            return []

        name_vectors = []
        labels = []
        for char in characters:
            token = char_id_to_token(char)
            if token in vectors:
                vec = np.array(vectors[token])
            else:
                vec = np.zeros(target_dim)
            if vec.size != target_dim:
                if vec.size == 0:
                    vec = np.zeros(target_dim)
                elif vec.size > target_dim:
                    vec = vec[:target_dim]
                else:
                    vec = np.pad(vec, (0, target_dim - vec.size), mode='constant')
            name_vectors.append(vec)
            labels.append(char)

        name_matrix = np.vstack(name_vectors)
        mean_vec = name_matrix.mean(axis=0)
        centered_matrix = name_matrix - mean_vec

        if len(name_vectors) < 2 or np.allclose(centered_matrix, centered_matrix[0]):
            pcs = np.zeros((len(name_vectors), 2))
        else:
            pcs = PCA(n_components=2).fit_transform(centered_matrix)

        x_vals = pcs[:, 0]
        y_vals = pcs[:, 1]
        norms = np.linalg.norm(centered_matrix, axis=1)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x_vals, y_vals, alpha=0.75, s=110, c=norms, cmap='viridis')
        try:
            from adjustText import adjust_text
            texts = []
            for x, y, label in zip(x_vals, y_vals, labels):
                t = plt.annotate(label, (x, y), fontsize=7, alpha=0.85, ha='center', va='bottom')
                texts.append(t)
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
        except ImportError:
            x_range = x_vals.max() - x_vals.min() or 1
            y_range = y_vals.max() - y_vals.min() or 1
            offset_dist = min(x_range, y_range) * 0.06
            for idx, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
                angle = (idx * 137.5) % 360
                ox = offset_dist * np.cos(np.radians(angle))
                oy = offset_dist * np.sin(np.radians(angle))
                plt.annotate(
                    label, (x + ox, y + oy), fontsize=6, alpha=0.7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7, lw=0.5)
                )

        plt.xlabel('PCA dimension 1 (mean-centered name vectors)', fontsize=12)
        plt.ylabel('PCA dimension 2 (mean-centered name vectors)', fontsize=12)
        plt.title(f'{play_name} - W2V Mean-Centered Name PCA', fontsize=14)
        plt.colorbar(scatter, label='Mean-centered vector L2 norm')
        plt.grid(True, alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Formula:}\ \tilde{v}_i = v_i - \bar{v},\ \mathrm{plot}\ \mathrm{PCA}_{1,2}(\tilde{v})$'
            + '\n'
            + rf'$\mathrm{{W2V:}}\ \mathrm{{window}}={window_size},\ \mathrm{{dim}}={vector_size}$',
            fontsize=9, ha='left', va='bottom'
        )
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        path = output_dir.rstrip('/') + '/'
        plt.savefig(os.path.join(path, f'{safe_name}_name_w2v_mean_centered_pca_scatter.svg'))
        plt.close()

        return [
            {'character': c, 'pc1': float(x), 'pc2': float(y), 'centered_norm': float(n)}
            for c, x, y, n in zip(labels, x_vals, y_vals, norms)
        ]
