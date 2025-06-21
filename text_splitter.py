import copy
import typing

from document import Document


class TextSplitter:
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200, add_index_in_metadata: bool = False) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f'Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller.'
            )
        self._chunk_size = chunk_size
        self._add_index_in_metadata = add_index_in_metadata
        self._chunk_overlap = chunk_overlap

    # Default split function (can be overridden in subclasses)
    def split_function(self, text: str) -> list[str]:
        return text.split()

    # Default join function (can be overridden in subclasses)
    def merge_function(self, parts: list[str]) -> str:
        return ' '.join(parts)

    def split_text(self, text: str) -> typing.List[str]:
        parts = self.split_function(text)
        chunks: typing.List[str] = []
        num_parts = len(parts)
        current_pos = 0

        while current_pos < num_parts:
            chunk_parts = parts[current_pos : current_pos + self._chunk_size]
            chunk = self.merge_function(chunk_parts)
            chunks.append(chunk)
            current_pos += self._chunk_size - self._chunk_overlap

        return chunks

    def split_documents(self, documents: typing.Iterable[Document]) -> typing.List[Document]:
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.content)
            metadatas.append(doc.metadata)
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_index_in_metadata:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata['start_index'] = index
                    previous_chunk_len = len(chunk)
                new_doc = Document(content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
