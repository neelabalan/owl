from urllib.parse import urlparse

import pydantic


class WebAddress(pydantic.BaseModel):
    url: str
    scheme: str = ''
    netloc: str = ''
    path: str = ''
    params: str = ''
    query: str = ''
    fragment: str = ''

    def __init__(self, url: str, **kwargs):
        # Parse immediately during initialization
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f'Invalid URL: {url}')

        super().__init__(
            url=url,
            scheme=parsed.scheme,
            netloc=parsed.netloc,
            path=parsed.path or '',
            params=parsed.params or '',
            query=parsed.query or '',
            fragment=parsed.fragment or '',
            **kwargs,
        )

    def __str__(self) -> str:
        return self.url

    @property
    def port(self) -> int | None:
        if ':' in self.netloc:
            try:
                return int(self.netloc.split(':')[1])
            except (ValueError, IndexError):
                return None
        return None

    class Config:
        frozen = True
