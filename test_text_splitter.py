import pathlib

import tiktoken

from document import DocumentLoader
from text_splitter import TextSplitter
from url import WebAddress


def direct_load():
    urls = [
        WebAddress(
            'https://raw.githubusercontent.com/kubernetes/website/refs/heads/main/content/en/blog/_posts/2015-09-00-Kubernetes-Performance-Measurements-And.md'
        ),
        WebAddress(
            'https://raw.githubusercontent.com/kubernetes/website/refs/heads/main/content/en/blog/_posts/2015-11-00-Creating-A-Raspberry-Pi-Cluster-Running-Kubernetes-The-Shopping-List-Part-1.md'
        ),
        WebAddress(
            'https://raw.githubusercontent.com/kubernetes/website/refs/heads/main/content/en/blog/_posts/2015-10-00-Some-Things-You-Didnt-Know-About-Kubectl_28.md'
        ),
    ]

    documents = [DocumentLoader.load(url) for url in urls]
    print(documents[0])
    print(documents[1])


def file_load():
    document = DocumentLoader.load(pathlib.Path('data/sample.txt'))
    text_splitter = TextSplitter(chunk_size=50, chunk_overlap=5)
    docs = text_splitter.split_documents([document])
    print(docs[0].content)
    print(docs[1].content)


class TokenTextSplitter(TextSplitter):
    def split_function(self, text: str) -> list[str]:
        encoder = tiktoken.encoding_for_model('gpt-4')
        print('got encoder')
        token_ids = encoder.encode(text)
        print('encoding done')
        return [encoder.decode([token_id]) for token_id in token_ids]

    def merge_function(self, parts):
        return ''.join(parts)


def file_load_token():
    document = DocumentLoader.load(pathlib.Path('data/sample.txt'))
    text_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=5)
    docs = text_splitter.split_documents([document])
    print(docs[0].content)
    print(docs[1].content)


file_load_token()
