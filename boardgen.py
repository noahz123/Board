import random

# Boggle dice for a 5x5 grid (same as in the JavaScript)
dice = [
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

def generate_board():
    # Shuffle the dice
    shuffled_dice = random.sample(dice, len(dice))
    
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
    with open(filename, 'w') as file:
        for row in board:
            file.write(','.join(row) + '\n')
    
    print(f"Board saved to {filename}")

if __name__ == "__main__":
    board = generate_board()
    save_board_to_file(board)
    
    # Print the generated board for verification
    print("Generated board:")
    for row in board:
        print(','.join(row))