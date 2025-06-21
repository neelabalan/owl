import mimetypes
import pathlib

import pydantic

# eventually get rid of requests as well if possible
# maybe httpx?
import requests

from owl import url

PathLike = str | pathlib.Path


class Document(pydantic.BaseModel):
    id: str | None = pydantic.Field(default=None)
    content: str | bytes
    source: url.WebAddress | pathlib.Path | None = None
    metadata: dict | pydantic.BaseModel | None = None
    model_config = pydantic.ConfigDict(frozen=True)

    def as_string(self):
        if isinstance(self.content, str):
            return self.content
        else:
            return self.content.decode()

    def as_bytes(self):
        if isinstance(self.content, bytes):
            return self.content
        else:
            return self.content.encode()

    def __str__(self) -> str:
        if self.metadata:
            return f"page_content='{self.content}' metadata={self.metadata}"
        return f"page_content='{self.content}'"


class DocumentLoader:
    @staticmethod
    def load(source: pathlib.Path | url.WebAddress) -> Document:
        if isinstance(source, pathlib.Path):
            return DocumentLoader.load_file(source)
        elif isinstance(source, url.WebAddress):
            return DocumentLoader.load_url(source)
        else:
            raise TypeError('source must be pathlib.Path or URL object')

    # TODO: add async
    @staticmethod
    def load_file(path: pathlib.Path) -> Document:
        content = path.read_text(encoding='utf-8')
        mime_type = mimetypes.guess_type(str(path))[0]
        file_type = path.suffix.lower().lstrip('.') or 'unknown'
        metadata = {
            'content_length': len(content),
            'file_size': path.stat().st_size,
            'file_type': file_type,
            'mime_type': mime_type,
            'source': f'file://{path.absolute()}',
        }
        return Document(content=content, source=path, metadata=metadata)

    @staticmethod
    def load_url(url: url.WebAddress) -> Document:
        resp = requests.get(str(url))
        resp.raise_for_status()
        content = resp.text
        mime_type = resp.headers.get('Content-Type', '')
        metadata = {
            'content_length': len(content),
            'mime_type': mime_type,
            'source': url.url,
            'response_headers': dict(resp.headers),
        }
        return Document(content=content, source=url, metadata=metadata)
