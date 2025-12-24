import xml.etree.ElementTree as ET
import BERT_Inference_Without_Finetune
from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                    'speech': "",
                }
                # Should be only one tei:ab in each sp tag
                for word in speech.find('tei:ab', XMLParser.namespaces):
                    speech_info['speech_text'] += word.text
                scene_speeches.append(speech_info)
            characters_speeches.append(scene_speeches)
        return characters_speeches # list(list(dict))

    
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
        cosine_similarity = {k: {k: 0 for k in self.characters} for k in self.characters}
        entities_contexts = {k: [] for k in self.characters}
        for scene in self.characters_speeches:
            for speech in scene:
                speaker = speech['speaker']
                speech_text = speech['speech']
                entities_contexts[speaker].append(speech_text)
        entities_embeddings_per_context, cls_per_context = BERT_Inference_Without_Finetune.inference_bert(entities_contexts)

        for speaker in self.characters:
            for other_speaker in self.characters:
                if speaker != other_speaker:
                    cosine_similarity[speaker][other_speaker] = 1 - spatial.distance.cosine(entities_embeddings_per_context[speaker], entities_embeddings_per_context[other_speaker])
        return cosine_similarity

    