from xmlparser import XMLParser

def main():
    xml_path = 'Data/Shakespeare/Hamlet/Hamlet.xml'
    parser = XMLParser(xml_path)
    parser.parse()
    parser.visualize_scatter()

if __name__ == "__main__":
    main()