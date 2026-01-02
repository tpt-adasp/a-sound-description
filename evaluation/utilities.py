# Utility functions to handle data


def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries in a way that extends parameters for common keys.
    """
    merged = dict1.copy()  # Make a copy of the first dictionary to preserve it
    for key, value in dict2.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                # If both values are dictionaries, recursively merge them
                merged[key] = merge_dicts(merged[key], value)
            else:
                # If not, overwrite the value
                merged[key] = value
        else:
            merged[key] = value
    return merged
