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

        self.characters = self.__get_characters()
        self.characters_speeches = self.__get_characters_speeches()

        self.co_occurrences = self.__calculate_co_occurrences()
        self.cosine_similarity = self.__calculate_cosine_similarity()

    

    def visualize_scatter(self):
        """
        Create a scatter plot: X = co-occurrences, Y = cosine similarity
        """
        # Prepare data
        pairs_data = []
        characters = list(self.co_occurrences.keys())
        
        for char1 in characters:
            for char2 in characters:
                if char1 != char2:
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


    def find_tag_occurrences(self, tag_name):
        """Find all elements with a specific tag name"""
        # Handle namespace
        results = self.root.findall(f'.//tei:{tag_name}', XMLParser.namespaces)
        return results
        
    def __get_characters(self):
        persons = self.find_tag_occurrences('person')
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
        
        for scene in scenes:
            scene_speeches = []
            for speech in scene.findall('tei:sp', XMLParser.namespaces):
                speech_info = {
                    'speaker': speech.get('who'),
                    'text': "",
                    'average_bert_embedding': None,
                }
                # Should be only one tei:ab in each sp tag
                for word in speech.find('tei:ab', XMLParser.namespaces):
                    speech_info['speech_text'] += word.text
                speech_info['average_bert_embedding'] = BERT_Inference_Without_Finetune.inference_bert_single(speech_info['text'])

                scene_speeches.append(speech_info)
            characters_speeches.append(scene_speeches)
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

    