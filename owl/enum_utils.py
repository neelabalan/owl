import enum


class LowercaseStringEnumMeta(type(enum.Enum)):
    """
    class Color(Enum, metaclass=LowercaseStringEnumMeta):
        RED = 1
        BLUE = 2

    print(Color._string_map)  # Should print {'red': <Color.RED: 1>, 'blue': <Color.BLUE: 2>}
    """

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        # Create the mapping dictionary
        cls._string_map = {}
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and isinstance(value, cls):
                cls._string_map[key.lower()] = value
