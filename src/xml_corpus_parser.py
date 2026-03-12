import os
import zipfile
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from xmlparser import XMLParser


class XMLCorpusParser:
    """
    Download Folger XML files, parse them with XMLParser, and aggregate
    cross-play pair differences between embedding sources.
    """

    def __init__(
        self,
        download_dir,
        options=None,
        xml_download_page="https://www.folgerdigitaltexts.org/download/xml.html",
        xml_urls=None,
    ):
        self.download_dir = download_dir
        self.xml_download_page = xml_download_page
        self.xml_urls = list(xml_urls) if xml_urls is not None else None
        self.options = options if options is not None else {"co-oc", "bert", "olmo"}
        self.parsers = {}

    def fetch_xml_urls(self):
        """
        Fetch a list of XML (or XML zip) download URLs from the Folger download page.
        If xml_urls were passed in the constructor, those are returned.
        """
        if self.xml_urls is not None:
            return list(self.xml_urls)

        response = requests.get(self.xml_download_page, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        urls = []
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            lower_href = href.lower()
            if lower_href.endswith(".xml") or lower_href.endswith(".zip"):
                urls.append(urljoin(self.xml_download_page, href))

        unique_urls = list(dict.fromkeys(urls))
        if not unique_urls:
            raise ValueError(
                "No XML or XML zip URLs found on the Folger download page. "
                "Provide xml_urls explicitly if the page structure changed."
            )
        return unique_urls

    def download_xml_files(self, force=False):
        """
        Download Folger XML files to the download directory.
        Supports both direct .xml links and .zip archives containing XML.
        """
        os.makedirs(self.download_dir, exist_ok=True)
        urls = self.fetch_xml_urls()

        for url in urls:
            filename = os.path.basename(urlparse(url).path)
            lower_name = filename.lower()
            if not (lower_name.endswith(".xml") or lower_name.endswith(".zip")):
                continue

            local_path = os.path.join(self.download_dir, filename)
            if os.path.exists(local_path) and not force and lower_name.endswith(".xml"):
                continue

            response = requests.get(url, timeout=60)
            response.raise_for_status()

            if lower_name.endswith(".zip"):
                zip_path = local_path
                if os.path.exists(zip_path) and not force:
                    # Still ensure extracted XMLs exist
                    self._extract_zip_xml(zip_path, force=force)
                    continue

                with open(zip_path, "wb") as file:
                    file.write(response.content)
                self._extract_zip_xml(zip_path, force=force)
            else:
                with open(local_path, "wb") as file:
                    file.write(response.content)

        return self.list_local_xml_files()

    def _extract_zip_xml(self, zip_path, force=False):
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                if not member.lower().endswith(".xml"):
                    continue
                target_path = os.path.join(self.download_dir, os.path.basename(member))
                if os.path.exists(target_path) and not force:
                    continue
                with archive.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

    def list_local_xml_files(self):
        if not os.path.isdir(self.download_dir):
            return []
        return [
            os.path.join(self.download_dir, fname)
            for fname in os.listdir(self.download_dir)
            if fname.lower().endswith(".xml")
        ]

    def load_parsers(self):
        """
        Create XMLParser instances for each local XML file and parse them.
        """
        self.parsers = {}
        for xml_path in self.list_local_xml_files():
            play_name = self._play_name_from_path(xml_path)
            parser = XMLParser(xml_path, options=self.options)
            parser.parse()
            self.parsers[play_name] = parser
        return self.parsers

    def find_top_n_pair_differences(
        self,
        n,
        embedding1_name="BERT",
        embedding2_name="OLMo",
        normalize_scope="per_model",
        output_csv_path=None,
    ):
        """
        Across all loaded plays, find the top-N character pairs with the
        largest normalized embedding difference.
        """
        all_rows = []
        for play_name, parser in self.parsers.items():
            df = self._compute_pair_differences(
                parser,
                embedding1_name=embedding1_name,
                embedding2_name=embedding2_name,
                normalize_scope=normalize_scope,
            )
            if not df.empty:
                df.insert(0, "play_name", play_name)
                all_rows.append(df)

        if not all_rows:
            print("Warning: No pair differences computed across plays.")
            return pd.DataFrame()

        full_df = pd.concat(all_rows, ignore_index=True)
        full_df = full_df.sort_values("normalized_difference", ascending=False)
        top_df = full_df.head(int(n))

        if output_csv_path is not None:
            top_df.to_csv(output_csv_path, index=False, encoding="utf-8")

        return top_df

    def plot_top_n_pair_differences(
        self,
        n,
        embedding1_name="BERT",
        embedding2_name="OLMo",
        normalize_scope="per_model",
        output_svg_path="output/AllPlays_BERT_vs_OLMo_sum_normalized_bar.svg",
    ):
        """
        Plot a bar chart for the top-N differences across all plays.
        """
        top_df = self.find_top_n_pair_differences(
            n=n,
            embedding1_name=embedding1_name,
            embedding2_name=embedding2_name,
            normalize_scope=normalize_scope,
        )
        if top_df.empty:
            return top_df

        labels = top_df["play_name"] + ":" + top_df["pair_label"]
        values = top_df["normalized_difference"].astype(float)

        plt.figure(figsize=(max(12, len(values) * 0.4), 8))
        bars = plt.bar(range(len(values)), values, alpha=0.8)
        colors = plt.cm.viridis((values - values.min()) / (values.max() - values.min() + 1e-10))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("Absolute difference (sum-normalized)")
        plt.xlabel("Play:Character Pair")
        plt.title(f"Top {n} {embedding1_name} vs {embedding2_name} differences across plays")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_svg_path)
        plt.show()

        return top_df

    def _compute_pair_differences(
        self,
        parser,
        embedding1_name,
        embedding2_name,
        normalize_scope,
    ):
        cos1 = self._get_cosine_map(parser, embedding1_name)
        cos2 = self._get_cosine_map(parser, embedding2_name)

        characters = list(cos1.keys())
        pairs_data = []
        for i, char1 in enumerate(characters):
            for char2 in characters[i + 1:]:
                pairs_data.append(
                    {
                        "char1": char1,
                        "char2": char2,
                        "pair_label": f"{char1}-{char2}",
                        "cosine_similarity_1": cos1[char1][char2],
                        "cosine_similarity_2": cos2[char1][char2],
                    }
                )

        df = pd.DataFrame(pairs_data)
        if df.empty:
            return df

        if normalize_scope not in {"per_model", "per_pair"}:
            raise ValueError("normalize_scope must be 'per_model' or 'per_pair'")

        if normalize_scope == "per_model":
            total1 = float(df["cosine_similarity_1"].sum())
            total2 = float(df["cosine_similarity_2"].sum())
            df["norm_1"] = 0.0 if total1 == 0 else df["cosine_similarity_1"] / total1
            df["norm_2"] = 0.0 if total2 == 0 else df["cosine_similarity_2"] / total2
        else:
            denom = df["cosine_similarity_1"] + df["cosine_similarity_2"]
            df["norm_1"] = df["cosine_similarity_1"] / denom.replace(0, pd.NA)
            df["norm_2"] = df["cosine_similarity_2"] / denom.replace(0, pd.NA)
            df["norm_1"] = df["norm_1"].fillna(0.0)
            df["norm_2"] = df["norm_2"].fillna(0.0)

        df["raw_difference"] = df["cosine_similarity_2"] - df["cosine_similarity_1"]
        df["normalized_difference"] = (df["norm_2"] - df["norm_1"]).abs()
        return df

    @staticmethod
    def _get_cosine_map(parser, embedding_name):
        name = embedding_name.lower()
        if "bert" in name:
            return parser.cosine_similarity_bert
        if "olmo" in name:
            return parser.cosine_similarity_olmo
        raise ValueError(f"Unknown embedding name: {embedding_name}")

    @staticmethod
    def _play_name_from_path(xml_path):
        base = os.path.basename(xml_path)
        return os.path.splitext(base)[0]

