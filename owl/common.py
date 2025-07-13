import datetime
import inspect
import json
import re
import typing

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime.datetime, datetime.date, datetime.time)):
            return o.isoformat()
        if isinstance(o, datetime.timedelta):
            return o.total_seconds()
        # Handle numpy types if numpy is available
        if HAS_NUMPY:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_)):
                return bool(o)
        return super().default(o)


def split_text_with_regex(
    text: str, separator: str, keep_separator: typing.Union[bool, typing.Literal['start', 'end']]
) -> list[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f'({separator})', text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == 'end'
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (splits + [_splits[-1]]) if keep_separator == 'end' else ([_splits[0]] + splits)
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != '']


def extract_function_description(func: typing.Callable) -> str:
    if not func.__doc__:
        return ''

    docstring = inspect.cleandoc(func.__doc__)
    lines = docstring.split('\n')

    description_lines = []

    for line in lines:
        stripped = line.strip()

        # Stop at any section marker (Args:, Returns:, Raises:, etc.)
        if stripped.lower() in [
            'args:',
            'arguments:',
            'parameters:',
            'params:',
            'returns:',
            'return:',
            'raises:',
            'yields:',
            'yield:',
            'examples:',
            'example:',
            'note:',
            'notes:',
        ]:
            break

        description_lines.append(line)

    return '\n'.join(description_lines).strip()
