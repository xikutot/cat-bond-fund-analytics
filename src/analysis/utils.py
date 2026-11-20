def expect[T](value: object, cls: type[T]) -> T:
    if not isinstance(value, cls):
        raise TypeError(f"expected {cls.__name__}, got {type(value).__name__}")
    return value
