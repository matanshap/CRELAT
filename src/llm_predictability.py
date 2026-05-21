"""
LLM Predictability Heatmap for Shakespeare Plays.

Two LLMs work together:
  Generator (Mistral-7B):           predicts the next character's response
                                     given the preceding speech.
  Evaluator (OLMo-Shakespeare):     measures semantic distance between the
                                     predicted response and the actual one.

For each directed character pair (X → Y) every semantic distance is
summed, then displayed as a heatmap.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from OLMo_Embeddings import OLMoModelManager
from xmlparser import XMLParser


def _get_device():
    """Pick the first working CUDA device, fall back to CPU."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.get_device_properties(i)
                return torch.device(f"cuda:{i}")
            except RuntimeError:
                continue
    return torch.device("cpu")

# ── configuration ─────────────────────────────────────────────────

GENERATOR_MODEL = "mistralai/Mistral-7B-v0.1"
OLMO_REPO_ID    = "mradermacher/OLMo-1B-Base-shakespeare-GGUF"
OLMO_FILENAME   = "OLMo-1B-Base-shakespeare.IQ3_M.gguf"
MAX_GEN_TOKENS  = 100
TOP_N           = 8
OUTPUT_DIR      = "output/"

PLAYS = [
    ("Data/XML/folger_corpus/Ham.xml",  "Hamlet"),
    ("Data/XML/folger_corpus/Lr.xml",   "King Lear"),
    ("Data/XML/folger_corpus/Oth.xml",  "Othello"),
    ("Data/XML/folger_corpus/Ado.xml",  "Much Ado About Nothing"),
    ("Data/XML/folger_corpus/AYL.xml",  "As You Like It"),
    ("Data/XML/folger_corpus/TN.xml",   "Twelfth Night"),
    ("Data/XML/folger_corpus/R2.xml",   "Richard II"),
    ("Data/XML/folger_corpus/R3.xml",   "Richard III"),
]


# ── helpers ───────────────────────────────────────────────────────

def get_top_speakers(parser, top_n=TOP_N):
    speech_counts = {char: 0 for char in parser.characters}
    for scene in parser.characters_speeches:
        for speech in scene:
            speaker = speech.get('speaker', '[UNKNOWN]')
            if speaker in speech_counts:
                speech_counts[speaker] += 1
    ranked = sorted(speech_counts.items(), key=lambda item: item[1], reverse=True)
    ranked = [item for item in ranked if item[0] != '[UNKNOWN]']
    return [char for char, _ in ranked[:top_n]]


def clean_name(char_id):
    """'Hamlet_Ham' → 'Hamlet'"""
    return char_id.split('_')[0]


def extract_consecutive_pairs(parser, characters):
    """All (speaker_a → speaker_b) consecutive turns among *characters*."""
    char_set = set(characters)
    pairs = []
    for scene in parser.characters_speeches:
        for i in range(len(scene) - 1):
            a, b = scene[i], scene[i + 1]
            if (a['speaker'] != b['speaker']
                    and a['speaker'] in char_set
                    and b['speaker'] in char_set):
                pairs.append({
                    'speaker_a': a['speaker'],
                    'text_a':    a['text'],
                    'speaker_b': b['speaker'],
                    'text_b':    b['text'],
                })
    return pairs


# ── Generator LLM ────────────────────────────────────────────────

class ResponseGenerator:
    """Predicts the next character's response using a causal language model."""

    def __init__(self, model_name=GENERATOR_MODEL):
        self.device = _get_device()
        print(f"Loading generator '{model_name}' on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Generator ready.")

    def predict(self, speaker_a, text_a, speaker_b, play_name=""):
        name_a = clean_name(speaker_a)
        name_b = clean_name(speaker_b)
        prompt = (
            f"From Shakespeare's \"{play_name}\".\n\n"
            f"{name_a}: {text_a[:600]}\n"
            f"{name_b}:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_GEN_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = out[0][inputs['input_ids'].shape[1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return gen_text.split('\n')[0].strip()


# ── Evaluator LLM ────────────────────────────────────────────────

class SemanticEvaluator:
    """Measures semantic distance between two texts via OLMo-Shakespeare embeddings."""

    def __init__(self, repo_id=OLMO_REPO_ID, filename=OLMO_FILENAME):
        self.olmo = OLMoModelManager.get_instance(
            repo_id=repo_id, filename=filename,
        )

    def embed_batch(self, texts):
        embeddings = []
        for t in texts:
            safe = t if t.strip() else "."
            embeddings.append(self.olmo.get_mean_pooled_embeddings(safe))
        return embeddings

    @staticmethod
    def distances(embeddings_a, embeddings_b):
        """Element-wise cosine distance  (1 − cosine_similarity)."""
        dists = []
        for ea, eb in zip(embeddings_a, embeddings_b):
            sim = torch.nn.functional.cosine_similarity(
                ea.unsqueeze(0), eb.unsqueeze(0),
            ).item()
            dists.append(1.0 - sim)
        return dists


# ── matrix construction ──────────────────────────────────────────

def build_predictability_matrix(pairs, generator, evaluator, characters,
                                play_name):
    n = len(characters)
    idx = {c: i for i, c in enumerate(characters)}
    dist_sum = np.zeros((n, n))
    count    = np.zeros((n, n), dtype=int)

    predictions = []
    print(f"  Generating {len(pairs)} predictions …")
    for p in tqdm(pairs, desc="  Generator", unit="pair"):
        pred = generator.predict(
            p['speaker_a'], p['text_a'], p['speaker_b'], play_name,
        )
        predictions.append(pred)

    actuals = [p['text_b'] for p in pairs]
    print("  Computing OLMo-Shakespeare embeddings for predictions and actuals …")
    emb_pred   = evaluator.embed_batch(predictions)
    emb_actual = evaluator.embed_batch(actuals)
    dists = evaluator.distances(emb_pred, emb_actual)

    for k, p in enumerate(pairs):
        i = idx[p['speaker_a']]
        j = idx[p['speaker_b']]
        dist_sum[i, j] += dists[k]
        count[i, j]    += 1

    return dist_sum, count, predictions, dists


# ── visualisation ────────────────────────────────────────────────

def plot_heatmap(matrix, count_matrix, characters, play_name,
                 output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    n = len(characters)
    labels = [clean_name(c) for c in characters]

    sym_sum   = matrix + matrix.T
    sym_count = count_matrix + count_matrix.T
    with np.errstate(divide='ignore', invalid='ignore'):
        normed = np.where(sym_count > 0, sym_sum / sym_count, 0)

    tri_mask = np.tri(n, dtype=bool)

    annot_strings = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if tri_mask[i, j] or sym_count[i, j] == 0:
                annot_strings[i, j] = ""
            else:
                annot_strings[i, j] = (
                    f"{normed[i, j]:.2f}\n({int(sym_count[i, j])})"
                )

    display_mask = tri_mask | (sym_count == 0)

    side = max(8, len(labels) * 1.1)
    fig, ax = plt.subplots(figsize=(side, side * 0.9))

    sns.heatmap(
        normed,
        xticklabels=labels,
        yticklabels=labels,
        annot=annot_strings,
        fmt='',
        cmap='YlOrRd',
        mask=display_mask,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Semantic distance  (1 − cos sim)'},
        annot_kws={'fontsize': 9},
    )

    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(
        f'{play_name} — LLM Predictability\n'
        f'Mean semantic distance per interaction  '
        f'(Mistral-7B predicted vs actual, OLMo-Shakespeare eval)',
        fontsize=13,
    )
    plt.tight_layout()

    safe = play_name.replace(' ', '_')
    path = os.path.join(output_dir, f'{safe}_predictability.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ── entry point ──────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="LLM predictability heatmap")
    ap.add_argument(
        '--play', type=str, default=None,
        help='Run only the play whose name contains this substring',
    )
    ap.add_argument(
        '--top-n', type=int, default=TOP_N,
        help='Number of top speakers to include',
    )
    args = ap.parse_args()

    plays = PLAYS
    if args.play:
        plays = [(p, n) for p, n in plays if args.play.lower() in n.lower()]
        if not plays:
            print(f"No play matching '{args.play}' found.")
            return

    generator = ResponseGenerator()
    evaluator = SemanticEvaluator()

    for xml_path, play_name in plays:
        print(f"\n{'=' * 60}")
        print(f"  {play_name}")
        print(f"{'=' * 60}")

        parser = XMLParser(xml_path, options={"co-oc"})
        parser.parse()

        chars = get_top_speakers(parser, top_n=args.top_n)
        pairs = extract_consecutive_pairs(parser, chars)
        print(f"  {len(pairs)} consecutive turns among top-{args.top_n} characters")

        if not pairs:
            print("  (no pairs — skipping)")
            continue

        dist_sum, count, predictions, dists = build_predictability_matrix(
            pairs, generator, evaluator, chars, play_name,
        )

        plot_heatmap(dist_sum, count, chars, play_name)

        safe = play_name.replace(' ', '_')

        txt_path = os.path.join(OUTPUT_DIR, f'{safe}_comparisons.txt')
        with open(txt_path, 'w') as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"  {play_name} — Predicted vs Actual Comparisons\n")
            f.write(f"  Generator: {GENERATOR_MODEL}\n")
            f.write(f"  Evaluator: {OLMO_REPO_ID}/{OLMO_FILENAME}\n")
            f.write(f"  {len(pairs)} interactions among top-{args.top_n} characters\n")
            f.write(f"{'=' * 80}\n\n")
            for k, p in enumerate(pairs):
                name_a = clean_name(p['speaker_a'])
                name_b = clean_name(p['speaker_b'])
                f.write(f"── Interaction {k+1}/{len(pairs)} "
                        f"({name_a} → {name_b})  "
                        f"distance = {dists[k]:.4f} ──\n\n")
                f.write(f"  {name_a} says:\n")
                f.write(f"    {p['text_a'][:300]}\n\n")
                f.write(f"  {name_b} actually says:\n")
                f.write(f"    {p['text_b'][:300]}\n\n")
                f.write(f"  Mistral predicted {name_b} would say:\n")
                f.write(f"    {predictions[k][:300]}\n\n")
        print(f"  → {txt_path}")

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_dist = np.where(count > 0, dist_sum / count, 0)

        data = {
            'play':          play_name,
            'characters':    chars,
            'labels':        [clean_name(c) for c in chars],
            'distance_sum':  dist_sum.tolist(),
            'count':         count.tolist(),
            'mean_distance': mean_dist.tolist(),
        }
        json_path = os.path.join(OUTPUT_DIR, f'{safe}_predictability.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  → {json_path}")


if __name__ == '__main__':
    main()
