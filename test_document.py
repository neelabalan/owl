import pathlib

import pytest

from document import Document
from document import DocumentLoader
from url import WebAddress


def test_document_as_string_returns_content():
    doc = Document(content='hello world')
    assert doc.as_string() == 'hello world'
    assert doc.as_bytes() == b'hello world'


def test_load_sample_files_from_data_dir():
    # Assume these files exist in ./data/
    data_dir = pathlib.Path(__file__).parent / 'data'
    files = [
        data_dir / 'sample.txt',
        data_dir / 'sample.json',
        data_dir / 'sample.csv',
    ]
    for file in files:
        doc = DocumentLoader.load_file(file)
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
        WebAddress('https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore'),
        WebAddress('https://raw.githubusercontent.com/typicode/json-server/master/package.json'),
        WebAddress('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'),
    ]
    for url in urls:
        doc = DocumentLoader.load_url(url)
        assert doc.source == url
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0
        meta = doc.metadata
        assert meta['mime_type'] is not None
        assert meta['content_length'] > 0
        assert meta['source'] == url.url
        assert meta['response_status'] == 200
