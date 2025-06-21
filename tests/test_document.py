import pathlib

import pytest

from owl import document
from owl import url


def test_document_as_string_returns_content():
    doc = document.Document(content='hello world')
    assert doc.as_string() == 'hello world'
    assert doc.as_bytes() == b'hello world'


def test_load_sample_files_from_data_dir():
    # Assume these files exist in ./data/
    data_dir = pathlib.Path('data')
    files = [
        data_dir / 'sample.txt',
        data_dir / 'sample.json',
        data_dir / 'sample.csv',
    ]
    for file in files:
        doc = document.DocumentLoader.load_file(file)
        assert doc.source == file
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0
        assert doc.metadata['file_type'] == file.suffix.lstrip('.').lower()
        assert doc.metadata['file_size'] > 0
        assert doc.metadata['content_length'] > 0
        assert doc.metadata['source'].startswith('file://')
        assert doc.metadata['mime_type'] is not None


def test_load_real_github_files_minimal():
    urls = [
        url.WebAddress('https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore'),
        url.WebAddress('https://raw.githubusercontent.com/typicode/json-server/master/package.json'),
        url.WebAddress('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'),
    ]
    for _url in urls:
        doc = document.DocumentLoader.load_url(_url)
        assert doc.source == _url
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0
        meta = doc.metadata
        assert meta['mime_type'] is not None
        assert meta['content_length'] > 0
        assert meta['source'] == _url.url