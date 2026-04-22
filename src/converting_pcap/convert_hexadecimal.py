# author: lucia pintor

def convert_hexadecimal(input_string):
    """
    This function converts a hexadecimal number into decimal
    :param input_string:
    :return:
    """

    if "0x" in input_string:
        # if it is a hexadecimal value, then convert it
        return int(input_string, 16)
    
    else:
        return input_string
    
def convert_hexadecimal_list(hex_list):
    for h_id, h in enumerate(hex_list):
        hex_list[h_id] = convert_hexadecimal(h)
    return hex_list