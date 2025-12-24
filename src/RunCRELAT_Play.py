from xmlparser import XMLParser

def main():
    xml_path = 'Data/XML/hamlet_XML_FolgerShakespeare/Ham.xml'
    parser = XMLParser(xml_path)
    parser.parse()
    characters = [
        'Hamlet_Ham', 
        'Gertrude_Ham', 
        'Claudius_Ham', 
        'Polonius_Ham', 
        'Ophelia_Ham', 
        'Horatio_Ham', 
        'Laertes_Ham',
    ]
    parser.visualize_scatter(characters)
    parser.visualize_cooc_minus_cosine(characters)

if __name__ == "__main__":
    main()