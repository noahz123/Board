def calculate_total_value(filename):
    """
    Reads a file containing words and their values in the format:
    word: value
    
    Returns the sum of all values.
    """
    total = 0
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split the line at the colon and extract the value
                parts = line.strip().split(':')
                if len(parts) == 2:
                    word = parts[0].strip()
                    try:
                        value = int(parts[1].strip())
                        total += value
                    except ValueError:
                        print(f"Warning: Could not parse value for '{word}'")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    
    return total

if __name__ == "__main__":
    filename = "solutions.txt"
    result = calculate_total_value(filename)
    
    if result is not None:
        print(f"Total value of all words: {result}")