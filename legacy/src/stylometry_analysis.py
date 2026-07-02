#!/usr/bin/env python3
"""
Stylometry analysis for Shakespeare plays.
Extracts stylometric features (pronouns, word lengths, etc.) from raw text
and plots them chronologically by genre, parallel to existing analyses.
"""

import os
import sys
import json
import argparse
import re
from collections import Counter
import numpy as np
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.tokenize import word_tokenize

# Import PLAYS and plotting function from the existing script
sys.path.insert(0, os.path.dirname(__file__))
from genre_analysis import PLAYS, plot_chronological_by_genre
from download_folger_raw_text import ensure_raw_text_exists

def _records_with_flipped_feature(records, feature):
    """Return plot-only records where the selected feature is multiplied by -1."""
    flipped = []
    for r in records:
        rec = dict(r)
        value = rec.get(feature, np.nan)
        if np.isfinite(value):
            rec[feature] = -float(value)
        flipped.append(rec)
    return flipped

def clean_folger_text(text: str) -> str:
    """Removes the Folger header/metadata from the raw text."""
    # The actual play usually starts around "ACT 1"
    # We find the first occurrence of ACT 1 or Act 1
    match = re.search(r'(ACT\s+1\b|Act\s+1\b)', text)
    if match:
        text = text[match.start():]
        
    # Some texts might have footers or license info at the very end
    # Typically Folger texts end with the end of the play or FINIS
    # but the bulk of the text is what matters for frequencies.
    return text

def compute_stylometry(text: str) -> dict:
    """
    Computes stylometric markers from the text.
    These features match standard digital humanities markers.
    """
    words = word_tokenize(text)
    # Filter out punctuation for word-level stats
    words_alpha = [w.lower() for w in words if w.isalpha()]
    
    if not words_alpha:
        return {}
        
    n_words = len(words_alpha)
    n_tokens_for_normalization = n_words / 1000.0  # Frequencies per 1000 words
    
    # Feature 1: Mean letters per word
    mean_letters = sum(len(w) for w in words_alpha) / n_words
    
    # Feature 2: Number of BWs (Bag of Words / Unique words / Types)
    types = set(words_alpha)
    unique_words_per_1k = len(types) / n_tokens_for_normalization
    
    # Pronoun and function word categories
    first_person_sing = {"i", "me", "my", "mine", "myself"}
    first_person_plur = {"we", "us", "our", "ours", "ourselves"}
    second_person_sing = {"thou", "thee", "thy", "thine", "thyself"}
    second_person_plur = {"you", "your", "yours", "yourself", "yourselves"}
    third_person = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves", "it", "its", "itself"}
    determiners = {"the", "a", "an"}
    prepositions = {"in", "of", "with", "at", "from", "by", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among"}
    conjunctions = {"and", "but", "or", "nor", "for", "yet", "so", "although", "because", "since", "unless", "if"}
    aux_verbs = {"is", "am", "are", "was", "were", "be", "being", "been", "has", "have", "had", "do", "does", "did", "can", "could", "shall", "should", "will", "would", "may", "might", "must"}
    negations = {"not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot", "can't", "don't", "won't", "isn't", "aren't"}
    
    counts = Counter(words_alpha)
    
    def count_set(word_set):
        return sum(counts[w] for w in word_set)
    
    # Analyze semantic categories with Empath
    from empath import Empath
    lexicon = Empath()
    
    empath_cats = ["positive_emotion", "negative_emotion", "cognitive", "social", "family"]
    # We join words_alpha back to a string for empath to analyze
    clean_text_alpha = " ".join(words_alpha)
    empath_results = lexicon.analyze(clean_text_alpha, categories=empath_cats, normalize=False)
    if empath_results is None:
        empath_results = {cat: 0 for cat in empath_cats}
        
    # Certainty is not a default Empath category, so we retain the custom list for it
    certainty_words = {"all", "ever", "must", "every", "always", "never", "certain", "sure", "absolutely", "indeed", "truth", "true"}

    # Calculate frequencies per 1000 tokens
    n_1st_sing = count_set(first_person_sing) / n_tokens_for_normalization
    n_1st_plur = count_set(first_person_plur) / n_tokens_for_normalization
    n_2nd_sing = count_set(second_person_sing) / n_tokens_for_normalization
    n_2nd_plur = count_set(second_person_plur) / n_tokens_for_normalization
    n_3rd = count_set(third_person) / n_tokens_for_normalization
    
    n_personal_pronouns = n_1st_sing + n_1st_plur + n_2nd_sing + n_2nd_plur + n_3rd
    
    return {
        "bws_types_per_1k": float(unique_words_per_1k),
        "mean_letters": float(mean_letters),
        "personal_pronouns": float(n_personal_pronouns),
        "pronoun_1st_sing": float(n_1st_sing),
        "pronoun_1st_plur": float(n_1st_plur),
        "pronoun_2nd_sing": float(n_2nd_sing),
        "pronoun_2nd_plur": float(n_2nd_plur),
        "pronoun_3rd": float(n_3rd),
        "determiners": float(count_set(determiners) / n_tokens_for_normalization),
        "prepositions": float(count_set(prepositions) / n_tokens_for_normalization),
        "conjunctions": float(count_set(conjunctions) / n_tokens_for_normalization),
        "aux_verbs": float(count_set(aux_verbs) / n_tokens_for_normalization),
        "negations": float(count_set(negations) / n_tokens_for_normalization),
        "positive_emotions": float(empath_results["positive_emotion"] / n_tokens_for_normalization),
        "negative_emotions": float(empath_results["negative_emotion"] / n_tokens_for_normalization),
        "certainty_words": float(count_set(certainty_words) / n_tokens_for_normalization),
        "cognitive_words": float(empath_results["cognitive"] / n_tokens_for_normalization),
        "social_words": float(empath_results["social"] / n_tokens_for_normalization),
        "family_words": float(empath_results["family"] / n_tokens_for_normalization)
    }


def _build_interactive_explorer(valid_records, features, genre_set, genres, y, out_dir, best_combo, best_acc):
    """
    Pre-compute decision tree accuracy + per-play predictions for every
    pair and triplet of features, then emit a single self-contained HTML
    dashboard where the user can pick features, toggle 2D/3D, and see
    the accuracy and misclassified plays update instantly.
    """
    from sklearn.tree import DecisionTreeClassifier
    from itertools import combinations
    import numpy as np
    import json as _json

    print("Pre-computing decision tree results for all feature pairs and triplets...")

    titles = [r["title"] for r in valid_records]
    # Build feature matrix
    feat_data = {f: [r[f] for r in valid_records] for f in features}

    # Pre-compute for all pairs and triplets
    results = {}  # key -> {acc, predictions}

    for size in (2, 3):
        for combo in combinations(features, size):
            X = np.column_stack([feat_data[f] for f in combo])
            dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
            dt.fit(X, y)
            acc = dt.score(X, y)
            preds = dt.predict(X).tolist()
            key = "|".join(sorted(combo))
            results[key] = {"acc": round(acc * 100, 2), "preds": preds, "degree": dt.get_depth()}

    print(f"  Pre-computed {len(results)} combinations.")

    # Also compute singles
    for f in features:
        X = np.column_stack([feat_data[f]])
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
        dt.fit(X, y)
        results[f] = {"acc": round(dt.score(X, y) * 100, 2), "preds": dt.predict(X).tolist(), "degree": dt.get_depth()}

    # Build the play data for JS
    plays_js = []
    for i, r in enumerate(valid_records):
        play = {"title": titles[i], "genre": genres[i], "genreIdx": int(y[i])}
        for f in features:
            play[f] = r[f]
        plays_js.append(play)

    # Feature options as JSON
    feature_options = _json.dumps(features)
    plays_json = _json.dumps(plays_js)
    results_json = _json.dumps(results)
    genre_set_json = _json.dumps(genre_set)
    best_combo_json = _json.dumps(list(best_combo))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Shakespeare Stylometry Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
         background: #1a1a2e; color: #eee; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:18px; padding:18px 24px;
               align-items:center; background:#16213e; border-bottom:2px solid #0f3460; }}
  .controls label {{ font-size:13px; color:#aaa; margin-bottom:2px; display:block; }}
  .controls select {{ padding:6px 10px; border-radius:6px; border:1px solid #0f3460;
                      background:#1a1a2e; color:#eee; font-size:14px; min-width:180px; }}
  .controls button {{ padding:8px 18px; border-radius:6px; border:none; cursor:pointer;
                      font-weight:600; font-size:14px; transition:0.2s; }}
  .btn-2d {{ background:#e94560; color:#fff; }}
  .btn-3d {{ background:#0f3460; color:#fff; }}
  .btn-2d:hover {{ background:#c73a52; }}
  .btn-3d:hover {{ background:#1a5276; }}
  .btn-active {{ outline:3px solid #fff; }}
  #accuracy-box {{ background:#0f3460; padding:10px 20px; border-radius:8px;
                   font-size:18px; font-weight:700; letter-spacing:0.5px; }}
  #accuracy-box span {{ color:#e94560; }}
  #plot {{ width:100%; height:calc(100vh - 120px); }}
  #failed-list {{ padding:6px 24px; font-size:13px; color:#e94560; min-height:20px; }}
</style>
</head>
<body>
<div class="controls">
  <div>
    <label>X Axis</label>
    <select id="feat-x"></select>
  </div>
  <div>
    <label>Y Axis</label>
    <select id="feat-y"></select>
  </div>
  <div id="z-container">
    <label>Z Axis (3D only)</label>
    <select id="feat-z"></select>
  </div>
  <div>
    <label>Dimensions</label>
    <div style="display:flex;gap:6px;">
      <button class="btn-2d" id="btn-2d" onclick="setDim(2)">2 D</button>
      <button class="btn-3d btn-active" id="btn-3d" onclick="setDim(3)">3 D</button>
    </div>
  </div>
  <div id="accuracy-box">Accuracy: <span id="acc-val">—</span></div>
</div>
<div id="failed-list"></div>
<div id="plot"></div>

<script>
const FEATURES = {feature_options};
const PLAYS    = {plays_json};
const RESULTS  = {results_json};
const GENRES   = {genre_set_json};
const BEST     = {best_combo_json};
const COLORS   = {{"comedy":"#2ca02c","tragedy":"#d62728","history":"#1f77b4"}};

let dim = 3;

// Populate dropdowns
function populateDropdowns() {{
  ["feat-x","feat-y","feat-z"].forEach(id => {{
    const sel = document.getElementById(id);
    sel.innerHTML = "";
    FEATURES.forEach(f => {{
      const opt = document.createElement("option");
      opt.value = f; opt.textContent = f;
      sel.appendChild(opt);
    }});
  }});
  // Set defaults to best combo
  if (BEST.length >= 2) {{
    document.getElementById("feat-x").value = BEST[0];
    document.getElementById("feat-y").value = BEST[1];
  }}
  if (BEST.length >= 3) {{
    document.getElementById("feat-z").value = BEST[2];
  }}
}}

function setDim(d) {{
  if (dim !== d) {{
    Plotly.purge('plot');
  }}
  dim = d;
  document.getElementById("btn-2d").classList.toggle("btn-active", d===2);
  document.getElementById("btn-3d").classList.toggle("btn-active", d===3);
  document.getElementById("z-container").style.display = d===3 ? "block" : "none";
  updatePlot();
}}

function getKey() {{
  const fx = document.getElementById("feat-x").value;
  const fy = document.getElementById("feat-y").value;
  
  if (dim === 1) return fx; // Not currently used but for completeness
  
  let selected = [fx, fy];
  if (dim === 3) {{
    selected.push(document.getElementById("feat-z").value);
  }}
  
  // Sort alphabetically to match the sorted keys in Python
  selected.sort();
  
  // Return null if there are duplicates (Decision Tree requires distinct features)
  const unique = new Set(selected);
  if (unique.size < selected.length) return null;
  
  return selected.join("|");
}}

function updatePlot() {{
  const fx = document.getElementById("feat-x").value;
  const fy = document.getElementById("feat-y").value;
  const fz = document.getElementById("feat-z").value;
  const key = getKey();
  const res = RESULTS[key];

  if (!res) {{
    document.getElementById("acc-val").textContent = "N/A (choose distinct features)";
    document.getElementById("failed-list").textContent = "";
    return;
  }}

  document.getElementById("acc-val").textContent = res.acc.toFixed(2) + "%";

  // Build traces
  const traces = [];
  const failed_plays = [];

  GENRES.forEach((g, gi) => {{
    const correct_idx = [];
    const failed_idx = [];
    PLAYS.forEach((p, i) => {{
      if (p.genre === g) {{
        if (res.preds[i] === p.genreIdx) correct_idx.push(i);
        else {{ failed_idx.push(i); failed_plays.push(p.title + " (" + g + " predicted as " + GENRES[res.preds[i]] + ")"); }}
      }}
    }});

    if (dim === 3) {{
      if (correct_idx.length) {{
        traces.push({{
          type: 'scatter3d', mode: 'markers+text',
          x: correct_idx.map(i => PLAYS[i][fx]),
          y: correct_idx.map(i => PLAYS[i][fy]),
          z: correct_idx.map(i => PLAYS[i][fz]),
          text: correct_idx.map(i => PLAYS[i].title),
          textposition: 'top center', textfont: {{size:9, color:COLORS[g]}},
          marker: {{size:7, color:COLORS[g], symbol:'circle', opacity:0.85,
                    line:{{width:1,color:'white'}}}},
          name: g + ' (✓)',
          hovertemplate: '<b>%{{text}}</b><br>' + fx + ': %{{x:.3f}}<br>' + fy + ': %{{y:.3f}}<br>' + fz + ': %{{z:.3f}}<br>Genre: ' + g + '<br>Prediction: Correct<extra></extra>'
        }});
      }}
      if (failed_idx.length) {{
        traces.push({{
          type: 'scatter3d', mode: 'markers+text',
          x: failed_idx.map(i => PLAYS[i][fx]),
          y: failed_idx.map(i => PLAYS[i][fy]),
          z: failed_idx.map(i => PLAYS[i][fz]),
          text: failed_idx.map(i => PLAYS[i].title),
          textposition: 'top center', textfont: {{size:10, color:'#fff'}},
          marker: {{size:12, color:COLORS[g], symbol:'x', opacity:1.0,
                    line:{{width:2,color:'black'}}}},
          name: g + ' (✗ FAILED)',
          hovertemplate: '<b>%{{text}}</b><br>' + fx + ': %{{x:.3f}}<br>' + fy + ': %{{y:.3f}}<br>' + fz + ': %{{z:.3f}}<br>Genre: ' + g + '<br>Prediction: FAILED<extra></extra>'
        }});
      }}
    }} else {{
      if (correct_idx.length) {{
        traces.push({{
          type: 'scatter', mode: 'markers+text',
          x: correct_idx.map(i => PLAYS[i][fx]),
          y: correct_idx.map(i => PLAYS[i][fy]),
          text: correct_idx.map(i => PLAYS[i].title),
          textposition: 'top center', textfont: {{size:9, color:COLORS[g]}},
          marker: {{size:10, color:COLORS[g], symbol:'circle', opacity:0.85}},
          name: g + ' (✓)',
        }});
      }}
      if (failed_idx.length) {{
        traces.push({{
          type: 'scatter', mode: 'markers+text',
          x: failed_idx.map(i => PLAYS[i][fx]),
          y: failed_idx.map(i => PLAYS[i][fy]),
          text: failed_idx.map(i => PLAYS[i].title),
          textposition: 'top center', textfont: {{size:10, color:'#fff'}},
          marker: {{size:14, color:COLORS[g], symbol:'x', opacity:1.0}},
          name: g + ' (✗ FAILED)',
        }});
      }}
    }}
  }});

  const layout = dim === 3 ? {{
    scene: {{ xaxis: {{title: fx}}, yaxis: {{title: fy}}, zaxis: {{title: fz}} }},
    title: 'Shakespeare Stylometry — Decision Tree (degree ' + res.degree + ') Accuracy: ' + res.acc.toFixed(2) + '%',
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#1a1a2e',
    font: {{color: '#eee'}},
    legend: {{font:{{size:12}}}},
    margin: {{l:10, r:10, b:10, t:50}},
  }} : {{
    xaxis: {{title: fx}}, yaxis: {{title: fy}},
    title: 'Shakespeare Stylometry — Decision Tree (degree ' + res.degree + ') Accuracy: ' + res.acc.toFixed(2) + '%',
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
    font: {{color: '#eee'}},
    legend: {{font:{{size:12}}}},
    margin: {{l:60, r:20, b:60, t:50}},
  }};

  Plotly.react('plot', traces, layout, {{responsive: true}});

  document.getElementById("failed-list").innerHTML = failed_plays.length
    ? "<b>Misclassified plays:</b> " + failed_plays.join(", ")
    : "<b>All plays classified correctly!</b>";
}}

populateDropdowns();
document.getElementById("feat-x").addEventListener("change", updatePlot);
document.getElementById("feat-y").addEventListener("change", updatePlot);
document.getElementById("feat-z").addEventListener("change", updatePlot);
setDim(3);
</script>
</body>
</html>"""

    html_path = os.path.join(out_dir, "stylometry_explorer.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved interactive explorer to {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Stylometric analysis of Shakespeare texts")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for plots")
    parser.add_argument("--data", type=str, default=None, help="JSON path to save/load features")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    data_path = args.data or os.path.join(out_dir, "stylometry_analysis_data.json")

    records = []
    
    print("Extracting stylometric features using standard NLP techniques...")
    for xml_path, title, genre, year in PLAYS:
        play_code = os.path.splitext(os.path.basename(xml_path))[0]
        try:
            txt_path = ensure_raw_text_exists(play_code)
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            clean_text = clean_folger_text(raw_text)
            features = compute_stylometry(clean_text)
            
            if features:
                rec = {
                    "title": title,
                    "genre": genre,
                    "year": year,
                    "model": "stylometry" # Dummy model to satisfy plotting logic
                }
                rec.update(features)
                records.append(rec)
                print(f"Processed {play_code}: {len(clean_text)} chars")
            else:
                print(f"Warning: No features extracted for {play_code}")
                
        except Exception as e:
            print(f"Error processing {play_code}: {e}")
            continue

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {data_path}")

    features_to_plot = [
        "bws_types_per_1k", "mean_letters", "personal_pronouns", 
        "pronoun_1st_sing", "pronoun_1st_plur", "pronoun_2nd_sing", "pronoun_2nd_plur", "pronoun_3rd",
        "determiners", "prepositions", "conjunctions", "aux_verbs", "negations",
        "positive_emotions", "negative_emotions", "certainty_words", "cognitive_words", "social_words", "family_words"
    ]

    best_acc = 0.0
    best_combo = None
    
    # Find the best combination of features for predicting genre
    try:
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from itertools import combinations
        import numpy as np
        import matplotlib.pyplot as plt

        print("\n--- Finding the best feature combination for Genre Accuracy ---")
        # Filter valid records
        valid_records = [r for r in records if "year" in r and "genre" in r and r.get("genre") in ["comedy", "tragedy", "history"]]
        if len(valid_records) > 0:
            genres = [r["genre"] for r in valid_records]
            years = [r["year"] for r in valid_records]
            genre_set = sorted(list(set(genres)))
            y = np.array([genre_set.index(g) for g in genres])

            best_dt = None

            # Test single features
            print("Evaluating single features...")
            for feat in features_to_plot:
                X = np.column_stack([[r[feat] for r in valid_records]])
                dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42)
                dt.fit(X, y)
                acc = dt.score(X, y)
                if acc > best_acc:
                    best_acc = acc
                    best_combo = (feat,)
                    best_dt = dt

            # Test pairs of features
            print("Evaluating pairs of features...")
            for combo in combinations(features_to_plot, 2):
                X = np.column_stack([[r[combo[0]] for r in valid_records], [r[combo[1]] for r in valid_records]])
                dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
                dt.fit(X, y)
                acc = dt.score(X, y)
                if acc > best_acc:
                    best_acc = acc
                    best_combo = combo
                    best_dt = dt
                    
            # Test triplets of features
            print("Evaluating triplets of features...")
            for combo in combinations(features_to_plot, 3):
                X = np.column_stack([[r[combo[0]] for r in valid_records], [r[combo[1]] for r in valid_records], [r[combo[2]] for r in valid_records]])
                dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
                dt.fit(X, y)
                acc = dt.score(X, y)
                if acc > best_acc:
                    best_acc = acc
                    best_combo = combo
                    best_dt = dt

            print(f"\nBest feature combination found (Exhaustive Search): {best_combo}")
            print(f"Accuracy: {best_acc * 100:.2f}%")
            print("-------------------------------------------------------------")

            # Visualize the best decision tree
            if best_dt is not None:
                # Flowchart representation
                plt.figure(figsize=(14, 10))
                feature_names = list(best_combo)
                plot_tree(best_dt, feature_names=feature_names, class_names=genre_set, filled=True, rounded=True, fontsize=10)
                plt.title(f"Optimal Decision Tree (degree {best_dt.get_depth()}) for Genre Classification (Accuracy: {best_acc*100:.2f}%)")
                tree_out = os.path.join(out_dir, "best_combination_tree.svg")
                plt.savefig(tree_out, bbox_inches='tight')
                plt.close()
                print(f"Saved optimal decision tree plot to {tree_out}")

                # Build interactive explorer HTML
                _build_interactive_explorer(valid_records, features_to_plot, genre_set, genres, y, out_dir, best_combo, best_acc)

    except ImportError:
        print("Dependencies for feature search not installed. Skipping.")

    # Export to CSV (Features as rows, Plays as columns, like the requested table format)
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        if not df.empty:
            # Set the title as the column header
            df.set_index('title', inplace=True)
            df_stylometry = df[features_to_plot].T
            
            # Append the best accuracy findings to the end of the CSV
            if best_combo is not None:
                # Add two new rows at the bottom
                df_stylometry.loc['--- Best Optimization ---'] = ""
                df_stylometry.loc['Best Features'] = ", ".join(best_combo)
                df_stylometry.loc['Best Accuracy'] = f"{best_acc * 100:.2f}%"

            csv_path = os.path.join(out_dir, "stylometry_results_by_play.csv")
            df_stylometry.to_csv(csv_path)
            print(f"Saved stylometric CSV table to {csv_path}")
    except ImportError:
        print("pandas is not installed. Skipping CSV generation.")
        
    print(f"Plotting {len(features_to_plot)} features individually...")
    for feature in features_to_plot:
        plot_chronological_by_genre(
            _records_with_flipped_feature(records, feature),
            out_dir,
            features=[feature],
            degree=4,
        )
    print(f"Done! Plots saved in {out_dir}")



if __name__ == "__main__":
    main()
