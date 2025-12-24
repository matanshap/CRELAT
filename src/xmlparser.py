import xml.etree.ElementTree as ET
import BERT_Inference_Without_Finetune
from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

class XMLParser:
    namespaces = {
        'tei': 'http://www.tei-c.org/ns/1.0',
        "xml": 'http://www.w3.org/XML/1998/namespace',
    }
    text = None
    entities = None
    def __init__(self, xml_path):
        self.xml_path = xml_path
    
    def parse(self):
        with open(self.xml_path, 'r') as file:
            self.xml = file.read()
            self.root = ET.fromstring(self.xml)

        # Initialize BERT model manager once (will be reused for all speeches)
        self.bert_manager = BERT_Inference_Without_Finetune.BERTModelManager.get_instance()

        self.characters = [*self.__get_characters(), '[UNKNOWN]']
        self.characters_speeches = self.__get_characters_speeches()

        self.co_occurrences = self.__calculate_co_occurrences()
        self.cosine_similarity = self.__calculate_cosine_similarity()

    

    def visualize_scatter(self, characters_filter=None):
        """
        Create a scatter plot: X = co-occurrences, Y = cosine similarity
        
        Args:
            characters_filter: Optional list of characters. If provided, only pairs
                              where both characters are in this list will be displayed.
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
        
        plt.xlabel('Co-occurrences', fontsize=12)
        plt.ylabel('Cosine Similarity (BERT)', fontsize=12)
        plt.title('Character Relationships: Co-occurrences vs Semantic Similarity', fontsize=14)
        plt.colorbar(scatter, label='Cosine Similarity')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['co_occurrence'], df['cosine_similarity'], 1)
        p = np.poly1d(z)
        plt.plot(df['co_occurrence'], p(df['co_occurrence']), "r--", alpha=0.8, label='Trend')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('output/cooc_vs_cosine_scatter.svg')
        plt.show()
        
        return df

    def visualize_cooc_minus_cosine(self, characters_filter=None):
        """
        Create a bar plot: X = pairs, Y = co-occurrence - cosine similarity
        
        Args:
            characters_filter: Optional list of characters. If provided, only pairs
                              where both characters are in this list will be displayed.
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
        plt.ylabel('Co-occurrence - Cosine Similarity', fontsize=12)
        plt.title('Difference: Co-occurrence vs Semantic Similarity by Pair', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('output/cooc_minus_cosine_bar.svg')
        plt.show()
        
        return df


    def find_tag_occurrences(self, tag_name):
        """Find all elements with a specific tag name"""
        # Handle namespace
        results = self.root.findall(f'.//tei:{tag_name}', XMLParser.namespaces)
        return results
        
    def __get_characters(self):
        persons = self.find_tag_occurrences('person')
        persons.extend(self.find_tag_occurrences('personGrp'))
        characters = []
        for person in persons:
            characters.append(person.get(f'{{{XMLParser.namespaces["xml"]}}}id'))
        return characters

    def __get_characters_speeches(self):
        """
        Returns list(list(dict))
        Each list(dict) contains the speeches of the characters in the scene
        Each dict contains the speaker's name and the speech text
        The list(list(dict)) contains all the scenes in the play
        """
        characters_speeches = []
        scenes = self.find_tag_occurrences('div2')
        
        # Collect all speeches first
        for scene in scenes:
            scene_speeches = []
            for speech in scene.findall('tei:sp', XMLParser.namespaces):
                who = speech.get('who') # Remove the # symbol from the speaker id
                speech_info = {
                    # TODO: Handle multiple speakers in a single speech
                    'speaker': who.split()[0][1:] if who is not None else '[UNKNOWN]', # Remove the # symbol from the speaker id
                    'text': "",
                    'average_bert_embedding': None,
                }
                # Should be only one tei:ab in each sp tag
                ab_element = speech.find('tei:ab', XMLParser.namespaces)
                if ab_element is not None:
                    # milestone_correspondence = speech.get('corresp').split(' ')
                    for word in ab_element:
                        speech_info['text'] += word.text if word.text is not None else ""
                scene_speeches.append(speech_info)
            characters_speeches.append(scene_speeches)
            
        speeches_texts = [speech['text'] for scene in characters_speeches for speech in scene]
        # Process BERT embeddings in batch for better performance
        embeddings = self.bert_manager.get_embeddings_batch(speeches_texts, batch_size=16)
        
        # Assign embeddings back to speeches
        idx = 0
        for scene in characters_speeches:
            for speech_info in scene:
                speech_info['average_bert_embedding'] = embeddings[idx]
                idx += 1
        
        return characters_speeches # list(list(dict))

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

    def __calculate_cosine_similarity(self):
        cosine_similarities = {k: {k: 0 for k in self.characters} for k in self.characters}

        for scene in self.characters_speeches:
            for speech_idx in range(len(scene) - 1):
                speaker = scene[speech_idx]['speaker']
                # speech = scene[speech_idx]['speech']

                next_speaker = scene[speech_idx + 1]['speaker']
                # next_speech = scene[speech_idx + 1]['speech']

                # TODO: Check if there is a character that speaks twice in a row
                if speaker != next_speaker:
                    import torch.nn.functional as F

                # If embeddings are 1D tensors, add a dimension for batch processing
                cosine_similarities[speaker][next_speaker] += F.cosine_similarity(
                    scene[speech_idx]['average_bert_embedding'].unsqueeze(0),
                    scene[speech_idx + 1]['average_bert_embedding'].unsqueeze(0)
                ).item()
                cosine_similarities[next_speaker][speaker] = cosine_similarities[speaker][next_speaker]

        return cosine_similarities

    