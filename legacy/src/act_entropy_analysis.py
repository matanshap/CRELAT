"""
Act-level Entropy Analysis for Shakespeare Plays.

Two entropy metrics computed per act:

  1. LLM Cross-Entropy  — average per-token negative log-likelihood of each
     actual speech under a causal LM (Mistral-7B), conditioned on the
     preceding speeches in the same act.  Lower → more predictable dialogue.

  2. Interaction Graph Entropy  — Shannon entropy of the directed co-presence
     interaction graph.  When character X speaks, a directed edge X → Y is
     added for every other character Y currently on stage.  The edge-weight
     distribution is then normalised and its entropy computed.
     Higher → more evenly distributed interactions.

Both are plotted on a dual-axis chart with acts on the X-axis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import xml.etree.ElementTree as ET
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── configuration ─────────────────────────────────────────────────

GENERATOR_MODEL = "mistralai/Mistral-7B-v0.1"
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

NS = {
    'tei': 'http://www.tei-c.org/ns/1.0',
    'xml': 'http://www.w3.org/XML/1998/namespace',
}

CONTEXT_WINDOW = 10       # max previous speeches kept as LLM context
MAX_SPEECH_CHARS = 200    # truncate individual speeches fed to the LLM


# ── helpers ───────────────────────────────────────────────────────

def clean_name(char_id):
    """'Hamlet_Ham' → 'Hamlet'"""
    return char_id.split('_')[0]


# ── XML parsing ───────────────────────────────────────────────────

def _parse_scene(div2):
    """
    Walk a <div2> scene element in document order, tracking stage
    entrances / exits so that every speech carries a snapshot of who is
    on stage at the time it is delivered.

    Returns list of dicts:
        {'speaker': str, 'text': str, 'on_stage': frozenset[str]}
    """
    on_stage = set()
    result = []

    def _apply_stage(elem):
        stype = elem.get('type', '')
        who_str = elem.get('who', '')
        who_list = [w.lstrip('#') for w in who_str.split() if w.strip()]
        if stype == 'entrance':
            on_stage.update(who_list)
        elif stype == 'exit':
            on_stage.difference_update(who_list)

    for elem in div2:
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if tag == 'stage':
            _apply_stage(elem)

        elif tag == 'sp':
            who = elem.get('who', '')
            speaker = who.split()[0].lstrip('#') if who else '[UNKNOWN]'

            text_parts = []
            ab = elem.find('tei:ab', NS)
            if ab is not None:
                for w in ab:
                    if w.text:
                        text_parts.append(w.text)
            text = ''.join(text_parts).strip()

            on_stage.add(speaker)

            if text:
                result.append({
                    'speaker': speaker,
                    'text': text,
                    'on_stage': frozenset(on_stage),
                })

            for sub in elem.findall('.//tei:stage', NS):
                _apply_stage(sub)

    return result


def parse_play_by_acts(xml_path):
    """
    Parse a Folger TEI XML and return a list of acts.

    Each act is::

        {'act_num': int,
         'scenes': [  # list of scenes
             [  # list of speeches
                 {'speaker': str, 'text': str, 'on_stage': frozenset},
                 ...
             ],
         ]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    acts = []
    for div1 in root.findall('.//tei:div1', NS):
        if div1.get('type') != 'act':
            continue
        act_num = int(div1.get('n', str(len(acts) + 1)))
        scenes = [_parse_scene(d2) for d2 in div1.findall('tei:div2', NS)]
        acts.append({'act_num': act_num, 'scenes': scenes})

    return acts


# ── Interaction-graph entropy ─────────────────────────────────────

def build_interaction_graph(act):
    """
    Directed edge weights for an act.

    When X speaks and Y is on stage (Y ≠ X), edge (X → Y) gets +1.
    Returns  {(speaker, other): weight, …}
    """
    edges = {}
    for scene in act['scenes']:
        for speech in scene:
            speaker = speech['speaker']
            for char in speech['on_stage']:
                if char != speaker:
                    key = (speaker, char)
                    edges[key] = edges.get(key, 0) + 1
    return edges


def graph_entropy(edges):
    """Shannon entropy (bits) of the normalised edge-weight distribution."""
    if not edges:
        return 0.0
    weights = np.array(list(edges.values()), dtype=float)
    total = weights.sum()
    if total == 0:
        return 0.0
    probs = weights / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ── LLM cross-entropy ────────────────────────────────────────────

def _get_device():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.get_device_properties(i)
                return torch.device(f"cuda:{i}")
            except RuntimeError:
                continue
    return torch.device("cpu")


class CrossEntropyCalculator:
    """Per-token cross-entropy of actual speech under a causal LM."""

    def __init__(self, model_name=GENERATOR_MODEL):
        self.device = _get_device()
        print(f"Loading model '{model_name}' on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model ready.")

    def speech_cross_entropy(self, context_str, speaker, actual_text):
        """
        Compute mean per-token cross-entropy (nats) of *actual_text*
        conditioned on *context_str* + speaker prefix.
        """
        prompt = f"{context_str}\n{clean_name(speaker)}:"
        full = f"{prompt} {actual_text}"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(
            full, add_special_tokens=False, truncation=True, max_length=1024,
        )

        if len(full_ids) <= len(prompt_ids):
            return None

        input_ids = torch.tensor([full_ids], device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits[0]   # (seq_len, vocab)

        start = max(len(prompt_ids) - 1, 0)
        end = len(full_ids) - 1
        if start >= end:
            return None

        shift_logits = logits[start:end]
        targets = torch.tensor(full_ids[start + 1: end + 1], device=self.device)

        loss = torch.nn.functional.cross_entropy(shift_logits, targets)
        return loss.item()


def compute_act_cross_entropy(act, calc, play_name=""):
    """Mean cross-entropy across all speeches in an act (skipping the first)."""
    speeches = [s for scene in act['scenes'] for s in scene]
    if len(speeches) < 2:
        return 0.0

    ces = []
    for i in tqdm(range(1, len(speeches)), desc="    CE", leave=False):
        ctx = speeches[max(0, i - CONTEXT_WINDOW):i]
        context_str = "\n".join(
            f"{clean_name(s['speaker'])}: {s['text'][:MAX_SPEECH_CHARS]}"
            for s in ctx
        )
        ce = calc.speech_cross_entropy(
            context_str, speeches[i]['speaker'],
            speeches[i]['text'][:MAX_SPEECH_CHARS],
        )
        if ce is not None:
            ces.append(ce)

    return float(np.mean(ces)) if ces else 0.0


# ── visualisation ─────────────────────────────────────────────────

def plot_dual_entropy(act_labels, graph_ents, llm_ces, play_name,
                      output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(len(act_labels))
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    c_graph = '#1976D2'
    c_llm   = '#D32F2F'

    # Graph entropy — left axis
    ln1 = ax1.plot(x, graph_ents, 'o-', color=c_graph, linewidth=2,
                   markersize=8, label='Interaction Graph Entropy')
    ax1.set_xlabel('Act', fontsize=13)
    ax1.set_ylabel('Graph Entropy (bits)', color=c_graph, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=c_graph)
    ax1.set_xticks(x)
    ax1.set_xticklabels(act_labels, fontsize=11)

    # LLM cross-entropy — right axis
    has_llm = any(v > 0 for v in llm_ces)
    if has_llm:
        ax2 = ax1.twinx()
        ln2 = ax2.plot(x, llm_ces, 's--', color=c_llm, linewidth=2,
                       markersize=8, label='LLM Cross-Entropy')
        ax2.set_ylabel('LLM Cross-Entropy (nats)', color=c_llm, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=c_llm)
        lines = ln1 + ln2
    else:
        lines = ln1

    ax1.legend(lines, [l.get_label() for l in lines],
               loc='upper left', fontsize=10)

    ax1.set_title(
        f'{play_name} — Act-level Entropy Analysis\n'
        'Interaction-graph entropy  vs  LLM speech cross-entropy',
        fontsize=13, fontweight='bold',
    )
    ax1.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    safe = play_name.replace(' ', '_')
    path = os.path.join(output_dir, f'{safe}_act_entropy.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")
    return path


# ── public entry point ────────────────────────────────────────────

def analyze_play(xml_path, play_name, calculator=None, output_dir=OUTPUT_DIR):
    """Run full act-level entropy analysis on one play."""
    print(f"\n{'═' * 60}")
    print(f"  {play_name}")
    print(f"{'═' * 60}")

    acts = parse_play_by_acts(xml_path)
    print(f"  {len(acts)} acts parsed")

    act_labels = []
    graph_ents = []
    llm_ces    = []

    for act in acts:
        label = f"Act {act['act_num']}"
        act_labels.append(label)

        edges = build_interaction_graph(act)
        ge = graph_entropy(edges)
        graph_ents.append(ge)
        n_speeches = sum(len(sc) for sc in act['scenes'])
        print(f"  {label}: {n_speeches} speeches, {len(edges)} directed edges, "
              f"graph H = {ge:.3f} bits")

        if calculator is not None:
            ce = compute_act_cross_entropy(act, calculator, play_name)
            llm_ces.append(ce)
            print(f"  {label}: LLM CE = {ce:.3f} nats")
        else:
            llm_ces.append(0.0)

    path = plot_dual_entropy(act_labels, graph_ents, llm_ces,
                             play_name, output_dir)

    safe = play_name.replace(' ', '_')
    data = {
        'play': play_name,
        'acts': act_labels,
        'graph_entropies': graph_ents,
        'llm_cross_entropies': llm_ces,
    }
    json_path = os.path.join(output_dir, f'{safe}_act_entropy.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  → {json_path}")

    return data


# ── CLI ───────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Act-level entropy analysis")
    ap.add_argument(
        '--play', type=str, default=None,
        help='Run only the play whose name contains this substring',
    )
    ap.add_argument(
        '--no-llm', action='store_true',
        help='Skip LLM cross-entropy (graph entropy only)',
    )
    args = ap.parse_args()

    plays = PLAYS
    if args.play:
        plays = [(p, n) for p, n in plays if args.play.lower() in n.lower()]
        if not plays:
            print(f"No play matching '{args.play}'.")
            return

    calculator = None if args.no_llm else CrossEntropyCalculator()

    for xml_path, play_name in plays:
        analyze_play(xml_path, play_name, calculator)


if __name__ == '__main__':
    main()
