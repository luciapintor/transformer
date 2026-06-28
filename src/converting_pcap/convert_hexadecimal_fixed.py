# author: lucia pintor
# modified: robust conversion for PyShark LayerFieldsContainer and mixed types


def convert_hexadecimal(input_value):
    """
    Convert a hexadecimal value to decimal when possible.

    The function is intentionally defensive because PyShark can return
    strings, integers, tuples, lists, LayerField objects or None depending
    on the PCAP and on the field parsed by tshark.
    """

    if input_value is None:
        return 0

    if isinstance(input_value, int):
        return input_value

    if isinstance(input_value, float):
        return input_value

    # PyShark fields are not always plain strings. Converting to str makes
    # checks such as "0x" safe and keeps the function from crashing.
    value = str(input_value).strip()

    if value == "" or value.lower() == "none":
        return 0

    try:
        if value.lower().startswith("0x"):
            return int(value, 16)

        # Some tshark values may arrive as hexadecimal strings without the
        # 0x prefix. This branch is conservative: it converts only strings
        # that clearly contain hex letters.
        if any(c in value.lower() for c in "abcdef") and all(c in "0123456789abcdef" for c in value.lower()):
            return int(value, 16)
    except (TypeError, ValueError):
        pass

    return value


def convert_hexadecimal_list(hex_values):
    """
    Convert a sequence of hexadecimal values into a normal Python list.

    This does not modify the input object in-place. That is important because
    PyShark may return a LayerFieldsContainer, which supports iteration but
    does not support item assignment.
    """

    if hex_values is None:
        return []

    if isinstance(hex_values, (str, int, float)):
        values = [hex_values]
    else:
        try:
            values = list(hex_values)
        except TypeError:
            values = [hex_values]

    return [convert_hexadecimal(value) for value in values]
