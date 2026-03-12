from xmlparser import XMLParser
from character_visualizer import CharacterVisualizer

def main():
    # xml_path_hamlet = 'Data/XML/hamlet_XML_FolgerShakespeare/Ham.xml'
    # # parser_hamlet = XMLParser(xml_path_hamlet)
    # # parser_hamlet.parse()
    # # characters = [
    # #     'Hamlet_Ham', 
    # #     'Gertrude_Ham', 
    # #     'Claudius_Ham', 
    # #     'Polonius_Ham', 
    # #     'Ophelia_Ham', 
    # #     'Horatio_Ham', 
    # #     'Laertes_Ham',
    # # ]
    # # visualizer = CharacterVisualizer(parser_hamlet.co_occurrences, parser_hamlet.cosine_similarity_olmo)
    # # play_name = 'Hamlet, OLMo'
    
    # # visualizer.visualize_scatter(play_name, characters)
    # # visualizer.visualize_cooc_minus_cosine(play_name, characters)
    # # visualizer.visualize_cosine_over_cooc(play_name, characters, min_cooc_threshold=3)

    # # xml_path_the_merchant_of_venice = 'Data/XML/the-merchant-of-venice_XML_FolgerShakespeare/MV.xml'
    # # parser_the_merchant_of_venice = XMLParser(xml_path_the_merchant_of_venice)
    # # parser_the_merchant_of_venice.parse()
    # # characters = [
    # #     'Antonio_MV',
    # #     'Shylock_MV',
    # #     'Portia_MV',
    # #     'Bassanio_MV',
    # # ]
    # # visualizer = CharacterVisualizer(parser_the_merchant_of_venice.co_occurrences, parser_the_merchant_of_venice.cosine_similarity_olmo)
    # # play_name = 'The Merchant of Venice, OLMo'
    # # visualizer.visualize_scatter(play_name, characters)
    # # visualizer.visualize_cooc_minus_cosine(play_name, characters)
    # # visualizer.visualize_cosine_over_cooc(play_name, characters, min_cooc_threshold=3)

    # # # Example: Compare BERT and OLMo embeddings for a play
    # # # First, parse with both embeddings enabled
    xml_path_comparison = 'Data/XML/hamlet_XML_FolgerShakespeare/Ham.xml'
    parser_comparison = XMLParser(xml_path_comparison, options={"co-oc", "bert"})
    parser_comparison.parse()
    characters_comparison = [
        'Hamlet_Ham', 
        'Gertrude_Ham', 
        'Claudius_Ham', 
        'Polonius_Ham', 
        'Ophelia_Ham', 
        'Horatio_Ham', 
        'Laertes_Ham',
    ]
    # # Create visualizer with BERT embeddings
    # visualizer_comparison = CharacterVisualizer(parser_comparison.co_occurrences, parser_comparison.cosine_similarity_bert)
    # # Compare with OLMo embeddings
    # # visualizer_comparison.visualize_population_comparison(
    # #     'Hamlet', 
    # #     parser_comparison.cosine_similarity_olmo,
    # #     embedding1_name='BERT',
    # #     embedding2_name='OLMo',
    # #     characters_filter=characters_comparison
    # # )

    # visualizer_comparison.visualize_embedding_bar_comparison_sum_normalized(
    #     play_name="Hamlet",
    #     cosine_similarity_2=parser_comparison.cosine_similarity_olmo,
    #     embedding1_name="BERT",
    #     embedding2_name="OLMo",
    #     characters_filter=characters_comparison,
    #     top_n=None,
    # )

    # New diagrams: BERT scatter (X=interactions, Y=BERT) and W2V scatter (X=interactions, Y=w2v)
    parser_comparison.plot_bert_interactions_scatter(
        play_name="Hamlet", output_dir="output/", characters_filter=characters_comparison, min_cooc_threshold=3
    )
    parser_comparison.plot_bert_interactions_scatter_normalized(
        play_name="Hamlet", output_dir="output/", characters_filter=characters_comparison, min_cooc_threshold=3
    )

    parser_with_w2v = XMLParser(xml_path_comparison, options={"co-oc", "w2v"})
    parser_with_w2v.parse()
    parser_with_w2v.plot_w2v_interactions_scatter(
        play_name="Hamlet", output_dir="output/", characters_filter=characters_comparison, min_cooc_threshold=3
    )



    def get_top_speakers(parser, top_n=8, include_unknown=False):
        speech_counts = {char: 0 for char in parser.characters}
        for scene in parser.characters_speeches:
            for speech in scene:
                speaker = speech.get('speaker', '[UNKNOWN]')
                if speaker in speech_counts:
                    speech_counts[speaker] += 1
        ranked = sorted(speech_counts.items(), key=lambda item: item[1], reverse=True)
        if not include_unknown:
            ranked = [item for item in ranked if item[0] != '[UNKNOWN]']
        return [char for char, _ in ranked[:top_n]]

    famous_plays = [
        ("Data/XML/folger_corpus/Ham.xml", "Hamlet"),
        ("Data/XML/folger_corpus/R2.xml", "Richard II"),
        ("Data/XML/folger_corpus/Lr.xml", "King Lear"),
        ("Data/XML/folger_corpus/Tro.xml", "Troilus and Cressida"),
        ("Data/XML/folger_corpus/Ado.xml", "Much Ado About Nothing"),
        ("Data/XML/folger_corpus/AYL.xml", "As You Like It"),
        ("Data/XML/folger_corpus/Wiv.xml", "The Merry Wives of Windsor"),
        ("Data/XML/folger_corpus/TN.xml", "Twelfth Night"),
    ]
    for xml_path, play_name in famous_plays:
        parser = XMLParser(xml_path, options={"co-oc", "bert"})
        parser.parse()
        top_characters = get_top_speakers(parser, top_n=8)
        # parser.plot_name_pca_vs_speeches(
        #     play_name=play_name,
        #     output_dir="output/",
        #     characters_filter=top_characters,
        # )
        parser.plot_bert_interactions_scatter_normalized(
            play_name=play_name, output_dir="output/", characters_filter=top_characters, min_cooc_threshold=3
        )

    

if __name__ == "__main__":
    main()