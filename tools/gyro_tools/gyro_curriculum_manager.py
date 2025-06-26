import os
import requests
import shutil
import tarfile
import zipfile
import bz2
import gzip
from pathlib import Path
import typing

CURRICULUM_RESOURCES = {
    "wordnet": {
        "name": "WordNet 3.1",
        "url": "https://wordnetcode.princeton.edu/3.1/WordNet-3.1.tar.gz",
        "type": "tar.gz",
        "handler": "ingest_wordnet",
    },
    "wiktionary": {
        "name": "English Wiktionary",
        "url": "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2",
        "type": "bz2",
        "handler": "ingest_wiktionary",
    },
    "simplewiki": {
        "name": "Simple English Wikipedia",
        "url": "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2",
        "type": "bz2",
        "handler": "ingest_simplewiki",
    },
    "udhr": {
        "name": "UDHR Multilingual",
        "url": "https://unicode.org/udhr/assemblies/udhr_txt.zip",
        "type": "zip",
        "handler": "ingest_udhr",
    },
    "tatoeba": {
        "name": "Tatoeba English Sentences",
        "url": "https://downloads.tatoeba.org/exports/sentences.tar.bz2",
        "type": "tar.bz2",
        "handler": "ingest_tatoeba",
    },
    "opensubtitles": {
        "name": "OpenSubtitles Sample",
        "url": "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.en.gz",
        "type": "gz",
        "handler": "ingest_opensubtitles",
    },
    "wikibooks": {
        "name": "English Wikibooks",
        "url": "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2",
        "type": "bz2",
        "handler": "ingest_wikibooks",
    },
    "wikisource": {
        "name": "English Wikisource",
        "url": "https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2",
        "type": "bz2",
        "handler": "ingest_wikisource",
    },
    # Gutenberg and Common Crawl omitted for now due to complexity
}


def download_resource(resource_key, dest_dir, progress_cb=None):
    info = CURRICULUM_RESOURCES[resource_key]
    url = info["url"]
    filename = url.split("/")[-1]
    dest_path = Path(dest_dir) / filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)
    return dest_path


def extract_resource(archive_path, extract_dir):
    ext = str(archive_path)
    if ext.endswith(".tar.gz") or ext.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    elif ext.endswith(".tar.bz2"):
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=extract_dir)
    elif ext.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zipf:
            zipf.extractall(path=extract_dir)
    elif ext.endswith(".bz2"):
        out_path = Path(extract_dir) / Path(archive_path).stem
        with bz2.open(archive_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(typing.cast(typing.BinaryIO, f_in), f_out)
        return out_path
    elif ext.endswith(".gz"):
        out_path = Path(extract_dir) / Path(archive_path).stem
        with gzip.open(archive_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(typing.cast(typing.BinaryIO, f_in), f_out)
        return out_path
    else:
        raise ValueError(f"Unknown archive type: {archive_path}")
    return extract_dir


# Ingestion handlers (stubs for now)
def ingest_wordnet(path):
    # TODO: Implement actual ingestion logic
    print(f"[INFO] Ingesting WordNet from {path}")
    return True


def ingest_wiktionary(path):
    print(f"[INFO] Ingesting Wiktionary from {path}")
    return True


def ingest_simplewiki(path):
    print(f"[INFO] Ingesting Simple English Wikipedia from {path}")
    return True


def ingest_udhr(path):
    print(f"[INFO] Ingesting UDHR from {path}")
    return True


def ingest_tatoeba(path):
    print(f"[INFO] Ingesting Tatoeba from {path}")
    return True


def ingest_opensubtitles(path):
    print(f"[INFO] Ingesting OpenSubtitles from {path}")
    return True


def ingest_wikibooks(path):
    print(f"[INFO] Ingesting Wikibooks from {path}")
    return True


def ingest_wikisource(path):
    print(f"[INFO] Ingesting Wikisource from {path}")
    return True


def ingest_resource(resource_key, dest_dir, progress_cb=None):
    print(f"[INFO] Downloading {resource_key}...")
    archive_path = download_resource(resource_key, dest_dir, progress_cb)
    print(f"[INFO] Extracting {archive_path}...")
    extracted = extract_resource(archive_path, dest_dir)
    handler_name = CURRICULUM_RESOURCES[resource_key]["handler"]
    handler = globals()[handler_name]
    print(f"[INFO] Ingesting {resource_key}...")
    return handler(extracted)
