import argparse
import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_URL = "https://www.folger.edu/explore/shakespeares-works/download/"


def fetch_xml_urls(download_page):
    response = requests.get(download_page, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    urls = []
    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        if href.lower().endswith(".xml"):
            urls.append(urljoin(download_page, href))

    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        raise ValueError(
            "No XML URLs found on the Folger download page. "
            "The page structure may have changed."
        )
    return unique_urls


def download_xml_files(download_page, output_dir, force=False):
    os.makedirs(output_dir, exist_ok=True)
    urls = fetch_xml_urls(download_page)

    downloaded = []
    skipped = []
    for url in urls:
        filename = os.path.basename(urlparse(url).path)
        if not filename.lower().endswith(".xml"):
            continue
        local_path = os.path.join(output_dir, filename)
        if os.path.exists(local_path) and not force:
            skipped.append(local_path)
            continue

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(local_path, "wb") as file:
            file.write(response.content)
        downloaded.append(local_path)

    return downloaded, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download Folger Shakespeare XML files."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Folger download page URL to scrape for XML links.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("Data", "folger_xml"),
        help="Directory to save downloaded XML files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing files.",
    )
    args = parser.parse_args()

    downloaded, skipped = download_xml_files(
        download_page=args.url,
        output_dir=args.output_dir,
        force=args.force,
    )

    print(f"Downloaded: {len(downloaded)}")
    print(f"Skipped: {len(skipped)}")
    if downloaded:
        print("Saved files:")
        for path in downloaded:
            print(f"- {path}")


if __name__ == "__main__":
    main()


