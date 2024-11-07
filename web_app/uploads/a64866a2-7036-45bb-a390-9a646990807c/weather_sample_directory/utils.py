def to_float(value):
    try:
        return float(value)
    except:
        # No specific exception handling, returns a string if conversion fails
        return "N/A"
