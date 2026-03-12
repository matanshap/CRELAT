from xml_corpus_parser import XMLCorpusParser


def main():
    corpus = XMLCorpusParser(
        download_dir="Data/XML/folger_corpus",
        options={"co-oc", "bert", "olmo"},
    )
    corpus.download_xml_files()
    corpus.load_parsers()

    # Generate top-N CSV and SVG across all plays
    top_n = 30
    corpus.find_top_n_pair_differences(
        n=top_n,
        output_csv_path="output/AllPlays_top_30_pairs.csv",
    )
    corpus.plot_top_n_pair_differences(n=top_n)


if __name__ == "__main__":
    main()

