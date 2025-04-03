#!/usr/bin/env python3
"""
Word Game Utility

A combined script that provides functionality to:
1. Generate a random game board
2. Solve an existing game board
3. Upload board and solutions to Firebase storage
"""

import concurrent.futures
import threading
import time
import os
import requests
import random
import argparse
from collections import defaultdict
import firebase_admin
from firebase_admin import credentials, storage
import datetime

# Letter scores mapping
LETTER_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
    'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Qu': 10, 'R': 1,
    'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
}

# Boggle dice for a 5x5 grid
DICE = [
    ['A', 'A', 'A', 'F', 'R', 'S'],
    ['A', 'A', 'E', 'E', 'E', 'E'],
    ['A', 'A', 'F', 'I', 'R', 'S'],
    ['A', 'D', 'E', 'N', 'N', 'N'],
    ['A', 'E', 'E', 'E', 'E', 'M'],
    ['A', 'E', 'E', 'G', 'M', 'U'],
    ['A', 'E', 'G', 'M', 'N', 'N'],
    ['A', 'F', 'I', 'R', 'S', 'Y'],
    ['B', 'J', 'K', 'Qu', 'X', 'Z'],
    ['C', 'C', 'N', 'S', 'T', 'W'],
    ['C', 'E', 'I', 'I', 'L', 'T'],
    ['C', 'E', 'I', 'L', 'P', 'T'],
    ['C', 'E', 'I', 'P', 'S', 'T'],
    ['D', 'H', 'H', 'N', 'O', 'T'],
    ['D', 'H', 'H', 'L', 'O', 'R'],
    ['D', 'H', 'L', 'N', 'O', 'R'],
    ['D', 'D', 'L', 'N', 'O', 'R'],
    ['E', 'I', 'I', 'I', 'T', 'T'],
    ['E', 'M', 'O', 'T', 'T', 'T'],
    ['E', 'N', 'S', 'S', 'S', 'U'],
    ['F', 'I', 'P', 'R', 'S', 'Y'],
    ['G', 'O', 'R', 'R', 'V', 'W'],
    ['H', 'I', 'P', 'R', 'R', 'Y'],
    ['N', 'O', 'O', 'T', 'U', 'W'],
    ['O', 'O', 'O', 'T', 'T', 'U']
]

# GENERATOR FUNCTIONS

def generate_board():
    """Generate a random 5x5 board using the standard dice."""
    # Shuffle the dice
    shuffled_dice = random.sample(DICE, len(DICE))
    
    # Create a 5x5 board
    board = []
    for i in range(5):
        row = []
        for j in range(5):
            die_index = i * 5 + j
            die = shuffled_dice[die_index]
            face_index = random.randint(0, 5)  # Random face from each die
            letter = die[face_index]
            row.append(letter)
        board.append(row)
    
    return board

def save_board_to_file(board, filename="board.txt"):
    """Save the generated board to a file."""
    with open(filename, 'w') as file:
        for row in board:
            file.write(','.join(row) + '\n')
    
    print(f"Board saved to {filename}")

# SOLVER FUNCTIONS

def load_board(file_path=None):
    """
    Load a board from a file. If file_path is None, try to load from 'board.txt'
    in the same directory as this script.
    
    Expected format:
    R,C,S,T,E
    W,S,H,D,V
    B,O,N,G,E
    C,H,T,E,P
    I,S,N,R,T
    """
    # Bonus positions (preserved from original layout)
    bonus_positions = {
        (0, 0): 'DW', (0, 4): 'DW', (4, 0): 'DW', (4, 4): 'DW',  # Double Word
        (2, 2): 'TW',  # Triple Word
        (0, 2): 'TL', (2, 0): 'TL', (2, 4): 'TL', (4, 2): 'TL',  # Triple Letter
        (1, 1): 'DL', (1, 3): 'DL', (3, 1): 'DL', (3, 3): 'DL'   # Double Letter
    }
    
    # Default board if file can't be loaded
    default_board = [
        [('R', 1, 'DW'), ('C', 3, None), ('S', 1, 'TL'), ('T', 1, None), ('E', 1, 'DW')],
        [('W', 4, None), ('S', 1, 'DL'), ('H', 4, None), ('D', 2, 'DL'), ('V', 4, None)],
        [('B', 3, 'TL'), ('O', 1, None), ('N', 1, 'TW'), ('G', 2, None), ('E', 1, 'TL')],
        [('C', 3, None), ('H', 4, 'DL'), ('T', 1, None), ('E', 1, 'DL'), ('P', 3, None)],
        [('I', 1, 'DW'), ('S', 1, None), ('N', 1, 'TL'), ('R', 1, None), ('T', 1, 'DW')],
    ]
    
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "board.txt")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) != 5:
                raise ValueError(f"Board should have exactly 5 rows, found {len(lines)}")
                
            board = []
            for row_idx, line in enumerate(lines):
                row_letters = line.split(',')
                
                if len(row_letters) != 5:
                    raise ValueError(f"Row {row_idx+1} should have exactly 5 letters, found {len(row_letters)}")
                
                board_row = []
                for col_idx, letter in enumerate(row_letters):
                    letter = letter.strip().upper()
                    if not letter or len(letter) != 1 and letter != 'QU':
                        raise ValueError(f"Invalid letter at row {row_idx+1}, column {col_idx+1}: '{letter}'")
                    
                    # Get letter score from the LETTER_SCORES dictionary
                    letter_score = LETTER_SCORES.get(letter[0], 1)  # Default to 1 if letter not found
                    
                    # Get bonus from the bonus_positions dictionary
                    bonus = bonus_positions.get((row_idx, col_idx), None)
                    
                    board_row.append((letter, letter_score, bonus))
                
                board.append(board_row)
            
            print(f"Successfully loaded board from {file_path}")
            return board
            
    except Exception as e:
        print(f"Error loading board from {file_path}: {e}")
        print("Using default board layout.")
        return default_board

def load_dictionary(file_path=None):
    """
    Load a dictionary of valid words from a file or URL.
    Priority:
    1. Use 'dictionary.txt' in the same directory as this script
    2. Use provided file_path if it exists
    3. Try to fetch from GitHub URL
    4. Fall back to built-in sample if all fail
    """
    # Try to load from dictionary.txt in the same directory first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dict_path = os.path.join(script_dir, "dictionary.txt")
    
    if os.path.exists(local_dict_path):
        try:
            with open(local_dict_path, 'r', encoding='utf-8') as f:
                words = {word.strip().lower() for word in f if len(word.strip()) >= 3}
                print(f"Loaded dictionary from local file: {local_dict_path}")
                return words
        except Exception as e:
            print(f"Error loading dictionary from same directory: {e}")
    
    # If local dictionary.txt doesn't exist, try the provided path
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = {word.strip().lower() for word in f if len(word.strip()) >= 3}
                print(f"Loaded dictionary from file: {file_path}")
                return words
        except Exception as e:
            print(f"Error loading dictionary file: {e}")
    
    # If no local files work, try GitHub URL
    github_url = "https://raw.githubusercontent.com/noahz123/Board/refs/heads/main/dictionary.txt"
    try:
        print(f"Attempting to download dictionary from: {github_url}")
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        words = {word.strip().lower() for word in response.text.splitlines() if len(word.strip()) >= 3}
        print(f"Successfully loaded dictionary from URL with {len(words)} words")
        return words
        
    except Exception as e:
        print(f"Error downloading dictionary from URL: {e}")
    
    # Fallback: Built-in sample dictionary
    print("Using built-in sample dictionary.")
    return {
        "bone", "bore", "born", "cent", "cite", "code", "cone", "cope", "core", "corn",
        "dive", "done", "dose", "dove", "edge", "edit", "gene", "gone", "hide", "hint",
        "view", "vine", "vise", "visor", "whid", "whit"
    }

def is_adjacent(pos1, pos2):
    """Check if two positions are adjacent (horizontally, vertically, or diagonally)."""
    row1, col1 = pos1
    row2, col2 = pos2
    return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1 and pos1 != pos2

def calculate_word_score(path, board):
    """Calculate the score for a word based on the game rules."""
    word_score = 0
    word_multiplier = 1
    
    # Calculate base score with letter bonuses
    for row, col in path:
        letter, letter_score, bonus = board[row][col]
        
        # Apply letter bonuses
        if bonus == 'DL':
            word_score += letter_score * 2
        elif bonus == 'TL':
            word_score += letter_score * 3
        else:
            word_score += letter_score
        
        # Track word multipliers
        if bonus == 'DW':
            word_multiplier *= 2
        elif bonus == 'TW':
            word_multiplier *= 3
    
    # Apply word multiplier
    word_score *= word_multiplier
    
    # Bonus for longer words
    word_length = len(path)
    length_bonus = 0
    if word_length >= 6:
        length_bonus = 10
    elif word_length == 5:
        length_bonus = 5
    elif word_length == 4:
        length_bonus = 2
        
    word_score += length_bonus
    
    return word_score

def can_form_word(word, letter_grid, letter_positions):
    """
    Check if a word can be formed on the board and find all valid paths.
    This uses a more robust BFS approach to ensure valid paths.
    """
    # Create a set of valid letters in the word to quickly check if all are on the board
    word_letters = set(word)
    
    # First, check if all letters in the word exist on the board
    for letter in word_letters:
        if letter not in letter_positions:
            return []
    
    # All possible paths for this word
    valid_paths = []
    
    # Define a recursive path-finding function using backtracking
    def find_path(index, path, used_positions):
        # If we've used all letters, we found a valid path
        if index == len(word):
            valid_paths.append(path.copy())
            return
        
        # Current letter to find
        current_letter = word[index]
        
        # If this is the first letter, try all positions
        if index == 0:
            for pos in letter_positions[current_letter]:
                path.append(pos)
                used_positions.add(pos)
                find_path(index + 1, path, used_positions)
                used_positions.remove(pos)
                path.pop()
        else:
            # For subsequent letters, only try adjacent positions
            last_pos = path[-1]
            for pos in letter_positions[current_letter]:
                # Skip if already used
                if pos in used_positions:
                    continue
                
                # Check if adjacent to last position
                if is_adjacent(last_pos, pos):
                    path.append(pos)
                    used_positions.add(pos)
                    find_path(index + 1, path, used_positions)
                    used_positions.remove(pos)
                    path.pop()
    
    # Start the search
    find_path(0, [], set())
    return valid_paths

def process_word_batch(word_batch, counter, total_words, lock, letter_grid, letter_positions, board):
    """Process a batch of words and find valid paths for each."""
    results = {}
    local_counter = 0
    
    for word in word_batch:
        local_counter += 1
        
        # Find all possible paths for this word
        paths = can_form_word(word, letter_grid, letter_positions)
        
        if paths:
            # Calculate scores for each path
            word_results = []
            for path in paths:
                score = calculate_word_score(path, board)
                word_results.append((path, score))
            
            # Keep the path with the highest score
            best_path, best_score = max(word_results, key=lambda x: x[1])
            results[word] = (best_path, best_score)
        
        # Update and display progress every 20 words
        if local_counter % 20 == 0:
            with lock:
                counter[0] += 20
                percentage = (counter[0] / total_words) * 100
                print(f"Progress: {counter[0]}/{total_words} words processed ({percentage:.1f}%)")
    
    # Add remaining words to counter
    with lock:
        remaining = local_counter % 20
        if remaining > 0:
            counter[0] += remaining
    
    return results

def find_words_parallel(dictionary, letter_grid, letter_positions, board, num_threads=None):
    """Find all valid words on the board using parallel processing."""
    if num_threads is None:
        # Use default of number of CPU cores
        num_threads = os.cpu_count() or 4
    
    print(f"Starting word search with {num_threads} threads...")
    
    # Convert dictionary to list for batch processing
    word_list = sorted(list(dictionary))  # Sort for consistent testing
    total_words = len(word_list)
    print(f"Dictionary contains {total_words} words to check")
    
    # Calculate batch size (aim for each thread to process around 100 words at a time)
    batch_size = max(1, min(100, total_words // (num_threads * 2)))
    
    # Split dictionary into batches
    batches = [word_list[i:i+batch_size] for i in range(0, total_words, batch_size)]
    print(f"Split into {len(batches)} batches of approximately {batch_size} words each")
    
    # Shared counter for progress tracking with lock for thread safety
    counter = [0]
    lock = threading.Lock()
    
    # Process batches in parallel
    start_time = time.time()
    all_results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all batches for processing
        future_to_batch = {
            executor.submit(process_word_batch, batch, counter, total_words, lock, 
                            letter_grid, letter_positions, board): batch 
            for batch in batches
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results = future.result()
                all_results.update(batch_results)
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Found {len(all_results)} valid words on the board")
    
    return all_results

def display_path(word, path, board):
    """Display the path of a word on the board."""
    path_str = " -> ".join([f"{board[r][c][0]}({r},{c})" for r, c in path])
    return f"{word}: {path_str}"

def display_results(found_words, board):
    """Display the results of the word search and save to a file."""
    # Sort words by score (highest first)
    sorted_words = sorted(found_words.items(), key=lambda x: x[1][1], reverse=True)
    
    # Print results
    print(f"Found {len(found_words)} valid words")
    print("=" * 40)
    
    for word, (path, score) in sorted_words:
        print(f"{word}: {score} points")
    
    # Print highest-scoring word
    if sorted_words:
        highest_word, (highest_path, highest_score) = sorted_words[0]
        print(f"\nHighest-scoring word: {highest_word} with {highest_score} points")
        print(f"Path: {display_path(highest_word, highest_path, board)}")
    
    # Write results to a file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "solutions.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for word, (path, score) in sorted_words:
                f.write(f"{word}: {score}\n")
        
        print(f"\nSolution list written to {output_file}")
    except Exception as e:
        print(f"\nError writing solutions to file: {e}")

def generate_and_save():
    """Generate a random board and save it to file."""
    board = generate_board()
    save_board_to_file(board)
    
    # Print the generated board for verification
    print("Generated board:")
    for row in board:
        print(','.join(row))

def solve_board(board_file=None, dict_file=None, threads=None):
    """Solve an existing board."""
    # Load board
    board = load_board(board_file)
    
    # Extract letter grid for easier access
    letter_grid = [[cell[0].lower() for cell in row] for row in board]
    
    # Create a letter-to-positions mapping for quick lookups
    letter_positions = defaultdict(list)
    for row in range(5):
        for col in range(5):
            letter_positions[letter_grid[row][col]].append((row, col))
    
    # Load dictionary
    dictionary = load_dictionary(dict_file)
    
    # Find all valid words on the board
    found_words = find_words_parallel(dictionary, letter_grid, letter_positions, board, threads)
    
    # Display results
    display_results(found_words, board)

def initialize_firebase():
    """Initialize Firebase app with credentials."""
    # Firebase configuration
    firebase_config = {
        "apiKey": "AIzaSyCgBulWvIiylEBGbsjh0E-iE4yJonbzOew",
        "authDomain": "babble-game.firebaseapp.com",
        "projectId": "babble-game",
        "storageBucket": "babble-game.firebasestorage.app",  # Use the exact bucket name from GS URL
        "messagingSenderId": "679961772786",
        "appId": "1:679961772786:web:a836cab9bfc84b3289466b",
        "measurementId": "G-SQ2BQMP637",
        "databaseURL": "https://babble-game-default-rtdb.firebaseio.com"  # Required by pyrebase
    }
    
    try:
        import pyrebase
        
        # Initialize Pyrebase with the provided config
        firebase = pyrebase.initialize_app(firebase_config)
        
        # Get storage reference
        return firebase.storage()
    except Exception as e:
        print(f"Error initializing Firebase: {str(e)}")
        return None

def upload_file_to_firebase(local_file_path, remote_file_path=None):
    """Upload a file to Firebase Storage."""
    try:
        # Get firebase storage
        firebase_storage = initialize_firebase()
        
        if not firebase_storage:
            print("Failed to initialize Firebase storage")
            return False
        
        # If remote path not specified, use the local filename
        if not remote_file_path:
            remote_file_path = os.path.basename(local_file_path)
        
        # Upload to the main file (without date) for easy access
        firebase_storage.child(remote_file_path).put(local_file_path)
        
        print(f"Uploaded {local_file_path} to Firebase Storage as {remote_file_path}")
        return True
    except Exception as e:
        print(f"Error uploading file to Firebase: {str(e)}")
        return False

def upload_board_and_solutions():
    """Upload board.txt and solutions.txt to Firebase Storage."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    board_path = os.path.join(script_dir, "board.txt")
    solutions_path = os.path.join(script_dir, "solutions.txt")
    
    # Upload board.txt
    if os.path.exists(board_path):
        upload_file_to_firebase(board_path, "board.txt")
    else:
        print(f"Warning: {board_path} does not exist. Cannot upload board file.")
    
    # Upload solutions.txt
    if os.path.exists(solutions_path):
        upload_file_to_firebase(solutions_path, "solutions.txt")
    else:
        print(f"Warning: {solutions_path} does not exist. Cannot upload solutions file.")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Word Game Utility')
    parser.add_argument('action', choices=['generate', 'solve', 'both', 'upload'], 
                        help='Action to perform: generate a board, solve a board, both, or upload files',
                        default='schedule', nargs='?')  # Changed default from 'both' to 'schedule'
    parser.add_argument('-b', '--board-file', type=str, help='Path to board file (for solve action)', default='board.txt')
    parser.add_argument('-d', '--dict-file', type=str, help='Path to dictionary file (for solve action)', default='dictionary.txt')
    parser.add_argument('-t', '--threads', type=int, help='Number of threads for solving (default: number of CPU cores)', default=None)
    parser.add_argument('-u', '--upload', action='store_true', dest='upload', 
                        help='Upload board and solutions to Firebase')
    
    args = parser.parse_args()
    args.upload = True if not hasattr(args, 'upload') else args.upload
    
    if args.action == 'generate' or args.action == 'both':
        generate_and_save()
        print()
    
    if args.action == 'solve' or args.action == 'both':
        solve_board(args.board_file, args.dict_file, args.threads)
    
    # Upload files to Firebase if enabled (default) or if action is 'upload'
    if args.upload or args.action == 'upload':
        print("\nUploading files to Firebase...")
        upload_board_and_solutions()

if __name__ == "__main__":
    main()