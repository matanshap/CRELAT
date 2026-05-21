import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CharacterVisualizer:
    """
    Class for visualizing character relationships based on co-occurrences and cosine similarity.
    """
    
    def __init__(self, co_occurrences, cosine_similarity):
        """
        Initialize the visualizer with co-occurrence and cosine similarity data.
        
        Args:
            co_occurrences: Dictionary of dictionaries mapping character pairs to co-occurrence counts
            cosine_similarity: Dictionary of dictionaries mapping character pairs to cosine similarity scores
        """
        self.co_occurrences = co_occurrences
        self.cosine_similarity = cosine_similarity

    def _normalize_values(self, values, method):
        """
        Normalize a list or array of values based on the specified method.
        
        Args:
            values: List or array of numeric values to normalize
            method: Normalization method. Options:
                - 'max': Divide by maximum value (scales to 0-1 if all values >= 0)
                - 'minmax': Min-max normalization (scales to 0-1 range)
                - 'sum': Divide by sum (normalizes to proportions)
                - None: No normalization, return original values
        
        Returns:
            Normalized array
        """
        values_array = np.array(values)
        
        if method is None:
            return values_array
        
        if method == 'max':
            max_val = np.max(np.abs(values_array))
            if max_val == 0:
                return values_array
            return values_array / max_val
        
        elif method == 'minmax':
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            if max_val == min_val:
                return values_array
            return (values_array - min_val) / (max_val - min_val)
        
        elif method == 'sum':
            sum_val = np.sum(np.abs(values_array))
            if sum_val == 0:
                return values_array
            return values_array / sum_val
        
        else:
            raise ValueError(f"Unknown normalization method: {method}. Must be one of: 'max', 'minmax', 'sum', None")

    def visualize_scatter(
        self,
        play_name,
        characters_filter=None,
        normalize_cooc=None,
        *,
        cosine_ylabel=None,
        colorbar_label=None,
        plot_title=None,
        output_path=None,
        output_dir="output",
        filename_suffix="cooc_vs_cosine_scatter.svg",
        show=True,
    ):
        """
        Create a scatter plot: X = co-occurrences, Y = cosine similarity
        
        Args:
            play_name: Name of the play. Will be included in the title and filename.
            characters_filter: Optional list of characters. If provided, only pairs
                              where both characters are in this list will be displayed.
            normalize_cooc: Normalization method for co-occurrences only (cosine similarity is already 
                          normalized). Options: 'max', 'minmax', 'sum', None (default: None)
            cosine_ylabel: Y-axis label (default: Cosine Similarity (BERT))
            colorbar_label: Colorbar label (default: Cosine Similarity)
            plot_title: Full plot title (default: derived from play_name)
            output_path: If set, save figure to this path (dirs created as needed).
            output_dir: Used with filename_suffix when output_path is None (default: output)
            filename_suffix: Filename part after sanitized play_name (default: cooc_vs_cosine_scatter.svg)
            show: If True, call plt.show() after save; if False, plt.close() (default: True)
        """
        # Prepare data
        pairs_data = []
        characters = list(self.co_occurrences.keys())
        
        # Filter characters if filter list is provided
        if characters_filter is not None:
            characters = [char for char in characters if char in characters_filter]
        
        # Only create unique pairs (char1 < char2) to avoid duplicates
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                cooc = self.co_occurrences[char1][char2]
                cosim = self.cosine_similarity[char1][char2]
                pairs_data.append({
                    'char1': char1,
                    'char2': char2,
                    'co_occurrence': cooc,
                    'cosine_similarity': cosim
                })
        
        df = pd.DataFrame(pairs_data)
        
        # Apply normalization to co-occurrences only (cosine similarity is already normalized)
        if normalize_cooc is not None:
            df['co_occurrence'] = self._normalize_values(df['co_occurrence'].values, normalize_cooc)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['co_occurrence'], df['cosine_similarity'], 
                            alpha=0.6, s=100, c=df['cosine_similarity'], 
                            cmap='viridis')
        
        # Add character names to the plot with smart positioning
        # Try to use adjustText for better label placement, fallback to systematic offset
        try:
            from adjustText import adjust_text
            texts = []
            for idx, row in df.iterrows():
                label = f"{row['char1']}-{row['char2']}"
                text = plt.annotate(label, (row['co_occurrence'], row['cosine_similarity']),
                                  fontsize=7, alpha=0.8, ha='center', va='bottom')
                texts.append(text)
            # Adjust text positions to avoid overlaps
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5))
        except ImportError:
            # Fallback: use systematic offset based on point position to reduce overlap
            x_range = df['co_occurrence'].max() - df['co_occurrence'].min()
            y_range = df['cosine_similarity'].max() - df['cosine_similarity'].min()
            
            for idx, row in df.iterrows():
                label = f"{row['char1']}-{row['char2']}"
                # Use position-based offset to create a pattern that reduces overlap
                # Offset varies based on index to create spacing
                angle = (idx * 137.5) % 360  # Golden angle for better distribution
                offset_dist = min(x_range, y_range) * 0.08
                offset_x = offset_dist * np.cos(np.radians(angle))
                offset_y = offset_dist * np.sin(np.radians(angle))
                
                plt.annotate(label, 
                           (row['co_occurrence'] + offset_x, row['cosine_similarity'] + offset_y),
                           fontsize=6, alpha=0.7, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7, lw=0.5))
        
        # Update labels based on normalization
        cooc_label = 'Co-occurrences'
        if normalize_cooc == 'max':
            cooc_label = 'Co-occurrences (normalized by max)'
        elif normalize_cooc == 'minmax':
            cooc_label = 'Co-occurrences (min-max normalized)'
        elif normalize_cooc == 'sum':
            cooc_label = 'Co-occurrences (normalized by sum)'
        
        if cosine_ylabel is None:
            cosine_ylabel = "Cosine Similarity (BERT)"
        if colorbar_label is None:
            colorbar_label = "Cosine Similarity"
        if plot_title is None:
            plot_title = (
                f"{play_name} - Character Relationships: Co-occurrences vs Semantic Similarity"
            )

        plt.xlabel(cooc_label, fontsize=12)
        plt.ylabel(cosine_ylabel, fontsize=12)
        plt.title(plot_title, fontsize=14)
        plt.colorbar(scatter, label=colorbar_label)
        plt.grid(True, alpha=0.3)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Formula:}\ x=\mathrm{co\_occurrence},\ y=\mathrm{cosine\_similarity}$',
            fontsize=9, ha='left', va='bottom'
        )
        
        # Add trend line
        if len(df) >= 2:
            z = np.polyfit(df["co_occurrence"], df["cosine_similarity"], 1)
            p = np.poly1d(z)
            plt.plot(df["co_occurrence"], p(df["co_occurrence"]), "r--", alpha=0.8, label="Trend")
            plt.legend()
        
        plt.tight_layout()
        # Sanitize play_name for filename (replace spaces and special chars)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        if output_path:
            filename = output_path
            out_dir = os.path.dirname(os.path.abspath(filename))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        else:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{safe_name}_{filename_suffix}")
        plt.savefig(filename)
        if show:
            plt.show()
        else:
            plt.close()

        return df

    def visualize_cooc_minus_cosine(self, play_name, characters_filter=None, normalize_before_diff=None, normalize_difference=None):
        """
        Create a bar plot: X = pairs, Y = co-occurrence - cosine similarity
        
        Args:
            play_name: Name of the play. Will be included in the title and filename.
            characters_filter: Optional list of characters. If provided, only pairs
                              where both characters are in this list will be displayed.
            normalize_before_diff: Normalization method applied to BOTH co-occurrences and cosine similarity
                                  before computing the difference. This ensures they're on the same scale.
                                  Options: 'max', 'minmax', 'sum', None (default: None)
                                  Recommended: 'minmax' or 'max' to put both metrics on 0-1 scale
            normalize_difference: Normalization method for the difference itself after computation.
                                Options: 'max', 'minmax', 'sum', None (default: None)
        """
        # Prepare data
        pairs_data = []
        characters = list(self.co_occurrences.keys())
        
        # Filter characters if filter list is provided
        if characters_filter is not None:
            characters = [char for char in characters if char in characters_filter]
        
        # Only create unique pairs (char1 < char2) to avoid duplicates
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                cooc = self.co_occurrences[char1][char2]
                cosim = self.cosine_similarity[char1][char2]
                pairs_data.append({
                    'char1': char1,
                    'char2': char2,
                    'co_occurrence': cooc,
                    'cosine_similarity': cosim,
                    'difference': cooc - cosim
                })
        
        df = pd.DataFrame(pairs_data)
        
        # Apply the same normalization to both metrics before computing difference
        # This ensures they're on the same scale (important since co-occurrences are counts 
        # and cosine similarity is already bounded)
        if normalize_before_diff is not None:
            df['co_occurrence'] = self._normalize_values(df['co_occurrence'].values, normalize_before_diff)
            df['cosine_similarity'] = self._normalize_values(df['cosine_similarity'].values, normalize_before_diff)
            # Recompute difference after normalization
            df['difference'] = df['co_occurrence'] - df['cosine_similarity']
        
        # Apply normalization to the difference itself if specified
        if normalize_difference is not None:
            df['difference'] = self._normalize_values(df['difference'].values, normalize_difference)
        
        # Sort by difference for better visualization
        df = df.sort_values('difference', ascending=False)
        
        # Create pair labels
        df['pair_label'] = df['char1'] + '-' + df['char2']
        
        # Create bar plot
        plt.figure(figsize=(max(12, len(df) * 0.3), 8))
        bars = plt.bar(range(len(df)), df['difference'], alpha=0.7)
        
        # Color bars based on difference value
        colors = plt.cm.RdYlGn_r((df['difference'] - df['difference'].min()) / 
                                (df['difference'].max() - df['difference'].min()))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Set x-axis labels
        plt.xticks(range(len(df)), df['pair_label'], rotation=45, ha='right', fontsize=8)
        plt.xlabel('Character Pairs', fontsize=12)
        
        # Update y-axis label based on normalization
        ylabel = 'Co-occurrence - Cosine Similarity'
        if normalize_before_diff is not None or normalize_difference is not None:
            norm_parts = []
            if normalize_before_diff is not None:
                norm_parts.append(f'inputs:{normalize_before_diff}')
            if normalize_difference is not None:
                norm_parts.append(f'diff:{normalize_difference}')
            ylabel = f'Co-occurrence - Cosine Similarity (norm: {", ".join(norm_parts)})'
        
        plt.ylabel(ylabel, fontsize=12)
        title = f'{play_name} - Difference: Co-occurrence vs Semantic Similarity by Pair'
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Formula:}\ y=\mathrm{co\_occurrence}-\mathrm{cosine\_similarity}$',
            fontsize=9, ha='left', va='bottom'
        )
        
        plt.tight_layout()
        # Sanitize play_name for filename (replace spaces and special chars)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f'output/{safe_name}_cooc_minus_cosine_bar.svg'
        plt.savefig(filename)
        plt.show()
        
        return df

    def visualize_cosine_over_cooc(self, play_name, characters_filter=None, normalize_ratio=None, min_cooc_threshold=0):
        """
        Create a bar plot: X = pairs, Y = cosine_similarity / co_occurrence
        
        This ratio shows semantic similarity per interaction, which can reveal pairs with
        high semantic alignment relative to their interaction frequency.
        
        Args:
            play_name: Name of the play. Will be included in the title and filename.
            characters_filter: Optional list of characters. If provided, only pairs
                              where both characters are in this list will be displayed.
            normalize_ratio: Normalization method for the ratio itself.
                            Options: 'max', 'minmax', 'sum', None (default: None)
            min_cooc_threshold: Minimum co-occurrence count to include a pair (default: 0).
                               Pairs with co_occurrence < threshold will be excluded to avoid
                               division by zero or misleading high ratios from rare interactions.
        """
        # Prepare data
        pairs_data = []
        characters = list(self.co_occurrences.keys())
        
        # Filter characters if filter list is provided
        if characters_filter is not None:
            characters = [char for char in characters if char in characters_filter]
        
        # Only create unique pairs (char1 < char2) to avoid duplicates
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                cooc = self.co_occurrences[char1][char2]
                cosim = self.cosine_similarity[char1][char2]
                
                # Skip pairs with co-occurrence below threshold (to avoid division by zero or noisy ratios)
                if cooc <= 0 or cooc < min_cooc_threshold:
                    continue
                
                ratio = cosim / cooc
                pairs_data.append({
                    'char1': char1,
                    'char2': char2,
                    'co_occurrence': cooc,
                    'cosine_similarity': cosim,
                    'ratio': ratio
                })
        
        if len(pairs_data) == 0:
            print(f"Warning: No pairs found with co-occurrence >= {min_cooc_threshold}")
            return pd.DataFrame()
        
        df = pd.DataFrame(pairs_data)
        
        # Apply normalization to the ratio if specified
        if normalize_ratio is not None:
            df['ratio'] = self._normalize_values(df['ratio'].values, normalize_ratio)
        
        # Sort by ratio for better visualization
        df = df.sort_values('ratio', ascending=False)
        
        # Create pair labels
        df['pair_label'] = df['char1'] + '-' + df['char2']
        
        # Create bar plot
        plt.figure(figsize=(max(12, len(df) * 0.3), 8))
        bars = plt.bar(range(len(df)), df['ratio'], alpha=0.7)
        
        # Color bars based on ratio value
        colors = plt.cm.viridis((df['ratio'] - df['ratio'].min()) / 
                               (df['ratio'].max() - df['ratio'].min() + 1e-10))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Set x-axis labels
        plt.xticks(range(len(df)), df['pair_label'], rotation=45, ha='right', fontsize=8)
        plt.xlabel('Character Pairs', fontsize=12)
        
        # Update y-axis label based on normalization
        ylabel = 'Cosine Similarity / Co-occurrence'
        if normalize_ratio is not None:
            if normalize_ratio == 'max':
                ylabel = 'Cosine Similarity / Co-occurrence (normalized by max)'
            elif normalize_ratio == 'minmax':
                ylabel = 'Cosine Similarity / Co-occurrence (min-max normalized)'
            elif normalize_ratio == 'sum':
                ylabel = 'Cosine Similarity / Co-occurrence (normalized by sum)'
        
        plt.ylabel(ylabel, fontsize=12)
        title = f'{play_name} - Semantic Similarity per Interaction (Cosine Similarity / Co-occurrence)'
        if min_cooc_threshold > 0:
            title += f' (min co-occurrence: {min_cooc_threshold})'
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Formula:}\ y=\mathrm{cosine\_similarity}/\mathrm{co\_occurrence}$',
            fontsize=9, ha='left', va='bottom'
        )
        
        plt.tight_layout()
        # Sanitize play_name for filename (replace spaces and special chars)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f'output/{safe_name}_cosine_over_cooc_bar.svg'
        plt.savefig(filename)
        plt.show()
        
        return df

    def visualize_population_comparison(self, play_name, cosine_similarity_2, embedding1_name='BERT', 
                                       embedding2_name='OLMo', characters_filter=None):
        """
        Create a population plot comparing two different embeddings (e.g., BERT vs OLMo).
        Shows scatter plot, histograms, and correlation statistics.
        
        Args:
            play_name: Name of the play. Will be included in the title and filename.
            cosine_similarity_2: Second dictionary of dictionaries mapping character pairs to cosine similarity scores
            embedding1_name: Name/label for the first embedding (default: 'BERT')
            embedding2_name: Name/label for the second embedding (default: 'OLMo')
            characters_filter: Optional list of characters. If provided, only pairs
                              where both characters are in this list will be displayed.
        
        Returns:
            DataFrame with the comparison data
        """
        # Prepare data
        pairs_data = []
        characters = list(self.co_occurrences.keys())
        
        # Filter characters if filter list is provided
        if characters_filter is not None:
            characters = [char for char in characters if char in characters_filter]
        
        # Only create unique pairs (char1 < char2) to avoid duplicates
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                cosim1 = self.cosine_similarity[char1][char2]
                cosim2 = cosine_similarity_2[char1][char2]
                pairs_data.append({
                    'char1': char1,
                    'char2': char2,
                    f'cosine_similarity_{embedding1_name.lower()}': cosim1,
                    f'cosine_similarity_{embedding2_name.lower()}': cosim2,
                    'difference': cosim1 - cosim2,
                    'absolute_difference': abs(cosim1 - cosim2)
                })
        
        df = pd.DataFrame(pairs_data)
        
        if len(df) == 0:
            print(f"Warning: No pairs found for comparison")
            return pd.DataFrame()
        
        # Calculate correlation
        col1 = f'cosine_similarity_{embedding1_name.lower()}'
        col2 = f'cosine_similarity_{embedding2_name.lower()}'
        correlation = df[col1].corr(df[col2])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main scatter plot (top-left, spanning 2x2)
        ax_scatter = fig.add_subplot(gs[0:2, 0:2])
        scatter = ax_scatter.scatter(df[col1], df[col2], 
                                   alpha=0.6, s=100, 
                                   c=df['absolute_difference'], 
                                   cmap='viridis')
        
        # Add diagonal line (y=x) for reference
        min_val = min(df[col1].min(), df[col2].min())
        max_val = max(df[col1].max(), df[col2].max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 
                       'r--', alpha=0.5, linewidth=1, label='y=x (perfect agreement)')
        
        # Add trend line
        z = np.polyfit(df[col1], df[col2], 1)
        p = np.poly1d(z)
        ax_scatter.plot(df[col1], p(df[col1]), "b--", alpha=0.8, 
                       label=f'Trend (slope={z[0]:.3f})')
        
        ax_scatter.set_xlabel(f'Cosine Similarity ({embedding1_name})', fontsize=12)
        ax_scatter.set_ylabel(f'Cosine Similarity ({embedding2_name})', fontsize=12)
        ax_scatter.set_title(f'{play_name} - Embedding Comparison: {embedding1_name} vs {embedding2_name}', 
                           fontsize=14, fontweight='bold')
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend()
        plt.colorbar(scatter, ax=ax_scatter, label='Absolute Difference')
        ax_scatter.text(
            0.02, 0.02,
            rf'$\mathrm{{Formula:}}\ x=\cos_{{\mathrm{{{embedding1_name}}}}},\ y=\cos_{{\mathrm{{{embedding2_name}}}}}$',
            transform=ax_scatter.transAxes, fontsize=9, ha='left', va='bottom'
        )
        
        # Add correlation text
        ax_scatter.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=ax_scatter.transAxes, fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Histogram for embedding 1 (top-right)
        ax_hist1 = fig.add_subplot(gs[0, 2])
        ax_hist1.hist(df[col1], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax_hist1.set_xlabel(f'Cosine Similarity ({embedding1_name})', fontsize=10)
        ax_hist1.set_ylabel('Frequency', fontsize=10)
        ax_hist1.set_title(f'{embedding1_name} Distribution', fontsize=11)
        ax_hist1.grid(True, alpha=0.3, axis='y')
        ax_hist1.axvline(df[col1].mean(), color='red', linestyle='--', 
                        label=f'Mean: {df[col1].mean():.3f}')
        ax_hist1.legend(fontsize=8)
        
        # Histogram for embedding 2 (middle-right)
        ax_hist2 = fig.add_subplot(gs[1, 2])
        ax_hist2.hist(df[col2], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax_hist2.set_xlabel(f'Cosine Similarity ({embedding2_name})', fontsize=10)
        ax_hist2.set_ylabel('Frequency', fontsize=10)
        ax_hist2.set_title(f'{embedding2_name} Distribution', fontsize=11)
        ax_hist2.grid(True, alpha=0.3, axis='y')
        ax_hist2.axvline(df[col2].mean(), color='red', linestyle='--', 
                        label=f'Mean: {df[col2].mean():.3f}')
        ax_hist2.legend(fontsize=8)
        
        # Difference histogram (bottom-right)
        ax_diff = fig.add_subplot(gs[2, 2])
        ax_diff.hist(df['difference'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax_diff.set_xlabel(f'Difference ({embedding1_name} - {embedding2_name})', fontsize=10)
        ax_diff.set_ylabel('Frequency', fontsize=10)
        ax_diff.set_title('Difference Distribution', fontsize=11)
        ax_diff.grid(True, alpha=0.3, axis='y')
        ax_diff.axvline(0, color='black', linestyle='-', linewidth=1)
        ax_diff.axvline(df['difference'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["difference"].mean():.3f}')
        ax_diff.legend(fontsize=8)
        ax_diff.text(
            0.02, 0.02,
            rf'$\mathrm{{Formula:}}\ \Delta=\cos_{{\mathrm{{{embedding1_name}}}}}-\cos_{{\mathrm{{{embedding2_name}}}}}$',
            transform=ax_diff.transAxes, fontsize=8, ha='left', va='bottom'
        )
        
        # Box plot comparison (bottom-left, spanning 2 columns)
        ax_box = fig.add_subplot(gs[2, 0:2])
        box_data = [df[col1], df[col2]]
        bp = ax_box.boxplot(box_data, labels=[embedding1_name, embedding2_name], 
                           patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        ax_box.set_ylabel('Cosine Similarity', fontsize=12)
        ax_box.set_title('Distribution Comparison', fontsize=12)
        ax_box.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text box
        stats_text = f'Statistics:\n'
        stats_text += f'{embedding1_name} - Mean: {df[col1].mean():.3f}, Std: {df[col1].std():.3f}\n'
        stats_text += f'{embedding2_name} - Mean: {df[col2].mean():.3f}, Std: {df[col2].std():.3f}\n'
        stats_text += f'Correlation: {correlation:.3f}\n'
        stats_text += f'Mean Absolute Difference: {df["absolute_difference"].mean():.3f}'
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        # Sanitize play_name for filename (replace spaces and special chars)
        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        embedding1_safe = embedding1_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        embedding2_safe = embedding2_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f'output/{safe_name}_population_comparison_{embedding1_safe}_vs_{embedding2_safe}.svg'
        plt.savefig(filename)
        plt.show()
        
        return df

    def visualize_embedding_bar_comparison_sum_normalized(
        self,
        play_name,
        cosine_similarity_2,
        embedding1_name='BERT',
        embedding2_name='OLMo',
        characters_filter=None,
        normalize_scope='per_model',
        sort_by='normalized_difference',
        top_n=None,
    ):
        """
        Create a grouped bar chart comparing two cosine-similarity sources per character pair.

        Normalization options:
        - per_model (default): normalize each embedding across all pairs so each model's bars sum to 1:
            norm1[p] = cos1[p] / sum_p cos1[p]
            norm2[p] = cos2[p] / sum_p cos2[p]
        - per_pair: normalize within each pair by the pairwise sum:
            norm1[p] = cos1[p] / (cos1[p] + cos2[p])
            norm2[p] = cos2[p] / (cos1[p] + cos2[p])

        Pairs are then sorted by descending difference (embedding2 - embedding1) by default.

        Args:
            play_name: Name of the play. Used in title and output filename.
            cosine_similarity_2: Second dict-of-dicts mapping char pairs to cosine similarity scores.
            embedding1_name: Label for the first embedding (default: 'BERT').
            embedding2_name: Label for the second embedding (default: 'OLMo').
            characters_filter: Optional list of characters to restrict the pairs.
            normalize_scope: 'per_model' or 'per_pair' (default: 'per_model').
            sort_by: Sorting key. Options:
                - 'normalized_difference' (default): sort by (norm2 - norm1) desc
                - 'raw_difference': sort by (cos2 - cos1) desc
                - 'embedding2': sort by cos2 desc
                - 'embedding1': sort by cos1 desc
            top_n: If provided, only plot the top N pairs after sorting.

        Returns:
            DataFrame with per-pair raw, normalized values, and differences.
        """
        pairs_data = []
        characters = list(self.co_occurrences.keys())

        if characters_filter is not None:
            characters = [char for char in characters if char in characters_filter]

        if normalize_scope not in {'per_model', 'per_pair'}:
            raise ValueError("normalize_scope must be one of: 'per_model', 'per_pair'")

        # Only create unique pairs (char1 < char2) to avoid duplicates
        for i, char1 in enumerate(characters):
            for char2 in characters[i + 1:]:
                cos1 = self.cosine_similarity[char1][char2]
                cos2 = cosine_similarity_2[char1][char2]

                pairs_data.append({
                    'char1': char1,
                    'char2': char2,
                    f'cosine_similarity_{embedding1_name.lower()}': cos1,
                    f'cosine_similarity_{embedding2_name.lower()}': cos2,
                    'raw_difference': cos2 - cos1,
                })

        df = pd.DataFrame(pairs_data)
        if len(df) == 0:
            print("Warning: No pairs found for embedding bar comparison")
            return pd.DataFrame()

        df['pair_label'] = df['char1'] + '-' + df['char2']

        if sort_by not in {'normalized_difference', 'raw_difference', 'embedding2', 'embedding1'}:
            raise ValueError(
                f"Unknown sort_by={sort_by}. Must be one of: "
                f"'normalized_difference', 'raw_difference', 'embedding2', 'embedding1'"
            )

        col1_raw = f'cosine_similarity_{embedding1_name.lower()}'
        col2_raw = f'cosine_similarity_{embedding2_name.lower()}'
        col1_norm = f'{embedding1_name.lower()}_sum_normalized'
        col2_norm = f'{embedding2_name.lower()}_sum_normalized'

        # Compute normalized values
        if normalize_scope == 'per_model':
            total1 = float(df[col1_raw].sum())
            total2 = float(df[col2_raw].sum())
            df[col1_norm] = 0.0 if total1 == 0 else df[col1_raw] / total1
            df[col2_norm] = 0.0 if total2 == 0 else df[col2_raw] / total2
        else:  # per_pair
            denom = df[col1_raw] + df[col2_raw]
            df[col1_norm] = np.where(denom == 0, 0.0, df[col1_raw] / denom)
            df[col2_norm] = np.where(denom == 0, 0.0, df[col2_raw] / denom)

        df['normalized_difference'] = np.abs(df[col2_norm] - df[col1_norm])

        if sort_by == 'normalized_difference':
            df = df.sort_values('normalized_difference', ascending=False)
        elif sort_by == 'raw_difference':
            df = df.sort_values('raw_difference', ascending=False)
        elif sort_by == 'embedding2':
            df = df.sort_values(col2_raw, ascending=False)
        elif sort_by == 'embedding1':
            df = df.sort_values(col1_raw, ascending=False)

        if top_n is not None:
            df = df.head(int(top_n))

        # Plot grouped bars
        x = np.arange(len(df))
        width = 0.42

        plt.figure(figsize=(max(14, len(df) * 0.35), 8))
        plt.bar(x - width / 2, df[col1_norm], width, label=f'{embedding1_name} (sum-normalized)', alpha=0.85)
        plt.bar(x + width / 2, df[col2_norm], width, label=f'{embedding2_name} (sum-normalized)', alpha=0.85)

        plt.xticks(x, df['pair_label'], rotation=45, ha='right', fontsize=8)
        plt.ylabel('Sum-normalized cosine similarity', fontsize=12)
        plt.xlabel('Character Pairs', fontsize=12)

        if normalize_scope == 'per_model':
            norm_label = 'sum-normalized per model'
        else:
            norm_label = 'sum-normalized per pair'

        title = f'{play_name} - {embedding1_name} vs {embedding2_name} ({norm_label})'
        if sort_by == 'normalized_difference':
            title += f' | sorted by ({embedding2_name} - {embedding1_name})'
        elif sort_by == 'raw_difference':
            title += f' | sorted by raw ({embedding2_name} - {embedding1_name})'
        plt.title(title, fontsize=14)
        plt.gcf().text(
            0.01, 0.01,
            r'$\mathrm{Formulas:}\ \mathrm{per\_model}\ \hat{c}=\frac{c}{\sum c},\ \mathrm{per\_pair}\ \hat{c}=\frac{c}{c_1+c_2}$',
            fontsize=9, ha='left', va='bottom'
        )

        plt.grid(True, alpha=0.3, axis='y')
        # If per_pair, values are in [0,1]; if per_model they are proportions that also lie in [0,1]
        ymax = max(float(df[col1_norm].max()), float(df[col2_norm].max()), 0.0)
        plt.ylim(0, min(1.0, ymax * 1.15 if ymax > 0 else 1.0))
        plt.legend()
        plt.tight_layout()

        safe_name = play_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        embedding1_safe = embedding1_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        embedding2_safe = embedding2_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f'output/{safe_name}_{embedding1_safe}_vs_{embedding2_safe}_sum_normalized_bar.svg'
        plt.savefig(filename)
        plt.show()

        # Keep consistent return ordering/columns for downstream inspection.
        return df[['char1', 'char2', 'pair_label', col1_raw, col2_raw, col1_norm, col2_norm,
                   'raw_difference', 'normalized_difference']]

