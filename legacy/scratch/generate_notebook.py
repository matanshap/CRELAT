import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cells definition
cells = []

# 1. Title
cells.append(nbf.v4.new_markdown_cell("# Shakespeare Speech Similarity Analysis\n"
                                     "This notebook analyzes the semantic similarity between consecutive speeches in the Folger Shakespeare corpus.\n"
                                     "We use BERT-based cosine similarity scores stored in `output/speech_interactions_bert.csv`."))

# 2. Imports
cells.append(nbf.v4.new_code_cell("import pandas as pd\n"
                                 "import matplotlib.pyplot as plt\n"
                                 "import seaborn as sns\n"
                                 "import sys\n"
                                 "import os\n\n"
                                 "# Add src to path to import metadata\n"
                                 "sys.path.append('src')\n"
                                 "from genre_analysis import PLAYS\n\n"
                                 "sns.set_theme(style='whitegrid', palette='muted')"))

# 3. Loading Data
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading and Preparation\n"
                                     "We load the interaction data and join it with genre/year metadata from the original corpus definition."))

cells.append(nbf.v4.new_code_cell("df = pd.read_csv('output/speech_interactions_bert.csv')\n\n"
                                 "# Create metadata dataframe\n"
                                 "play_metadata = pd.DataFrame(PLAYS, columns=['path', 'title', 'genre', 'year'])\n\n"
                                 "# Merge metadata\n"
                                 "df = df.merge(play_metadata[['title', 'genre', 'year']], left_on='play', right_on='title', how='left')\n"
                                 "df.drop(columns=['title'], inplace=True)\n\n"
                                 "print(f'Loaded {len(df)} interactions across {df[\"play\"].nunique()} plays.')\n"
                                 "df.head()"))

# 4. Distribution
cells.append(nbf.v4.new_markdown_cell("## 2. Global Distribution of Similarity\n"
                                     "How similar are consecutive speeches in general?"))

cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(10, 5))\n"
                                 "sns.histplot(df['cosine_similarity'], bins=100, kde=True, color='teal')\n"
                                 "plt.title('Distribution of Cosine Similarity between Consecutive Speeches')\n"
                                 "plt.xlabel('Cosine Similarity Score')\n"
                                 "plt.ylabel('Frequency')\n"
                                 "plt.show()\n\n"
                                 "print('Summary Statistics:')\n"
                                 "display(df['cosine_similarity'].describe())"))

# 5. Top Similarities
cells.append(nbf.v4.new_markdown_cell("## 3. The Most Semantically Similar Conversations\n"
                                     "These pairs of speeches have the highest semantic overlap. This often happens in repetitive dialogue, shared metaphors, or when characters echo each other's words."))

cells.append(nbf.v4.new_code_cell("# Display top 10 most similar interactions\n"
                                 "top_similar = df.sort_values('cosine_similarity', ascending=False).head(10)\n\n"
                                 "for i, row in top_similar.iterrows():\n"
                                 "    print(f\"--- Rank {i+1} | Score: {row['cosine_similarity']:.4f} ---\")\n"
                                 "    print(f\"Play: {row['play']} | Scene: {row['scene']}\")\n"
                                 "    print(f\"[{row['speaker1']}]: {row['text1']}\")\n"
                                 "    print(f\"[{row['speaker2']}]: {row['text2']}\")\n"
                                 "    print('\\n')"))

# 6. Least Similarities
cells.append(nbf.v4.new_markdown_cell("## 4. The Most Semantically Distinct Transitions\n"
                                     "These pairs show the greatest semantic shift between speakers. This might indicate abrupt topic changes, misunderstandings, or formal shifts (e.g., from prose to verse or from one character's internal monologue to another's external interruption)."))

cells.append(nbf.v4.new_code_cell("# Display top 10 least similar interactions\n"
                                 "bottom_similar = df.sort_values('cosine_similarity', ascending=True).head(10)\n\n"
                                 "for i, row in bottom_similar.iterrows():\n"
                                 "    print(f\"--- Rank {i+1} | Score: {row['cosine_similarity']:.4f} ---\")\n"
                                 "    print(f\"Play: {row['play']} | Scene: {row['scene']}\")\n"
                                 "    print(f\"[{row['speaker1']}]: {row['text1']}\")\n"
                                 "    print(f\"[{row['speaker2']}]: {row['text2']}\")\n"
                                 "    print('\\n')"))

# 7. Genre Comparison
cells.append(nbf.v4.new_markdown_cell("## 5. Genre-based Comparison\n"
                                     "Does dialogue in Tragedies tend to be more or less semantically coherent than in Comedies?"))

cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(10, 6))\n"
                                 "sns.boxplot(x='genre', y='cosine_similarity', data=df, palette='viridis')\n"
                                 "plt.title('Speech Similarity by Genre')\n"
                                 "plt.show()\n\n"
                                 "print('Mean Similarity by Genre:')\n"
                                 "display(df.groupby('genre')['cosine_similarity'].mean().sort_values(ascending=False))"))

# 8. Chronological Trend
cells.append(nbf.v4.new_markdown_cell("## 6. Chronological Trend\n"
                                     "Does Shakespeare's dialogue semantic coherence change over time?"))

cells.append(nbf.v4.new_code_cell("play_avg = df.groupby(['play', 'year', 'genre'])['cosine_similarity'].mean().reset_index()\n\n"
                                 "plt.figure(figsize=(12, 6))\n"
                                 "sns.regplot(x='year', y='cosine_similarity', data=play_avg, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})\n"
                                 "plt.title('Evolution of Speech Similarity over Time')\n"
                                 "plt.xlabel('Year of Composition')\n"
                                 "plt.ylabel('Average Cosine Similarity')\n"
                                 "plt.show()"))

nb['cells'] = cells

with open('speech_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated: speech_analysis.ipynb")
