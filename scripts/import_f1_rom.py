"""
Script to import the F1 World Champion ROM into Gym Retro
"""

import retro
import os

def main():
    # Print the current list of available games
    print("Available games before import:")
    games = retro.data.list_games()
    print(games)
    
    # Get user input for ROM directory
    rom_path = input("Enter the full path to your F1 World Champion ROM file: ")
    
    if not os.path.exists(rom_path):
        print(f"Error: File {rom_path} does not exist.")
        return
    
    # Import the ROM
    print(f"Importing ROM from {rom_path}...")
    result = retro.data.add_rom_path(rom_path)
    
    if result:
        print("ROM imported successfully!")
    else:
        print("Failed to import ROM. Make sure the file is a valid ROM.")
    
    # Check if the game is now available
    print("\nAvailable games after import:")
    games = retro.data.list_games()
    print(games)
    
    # Check for F1 related games
    f1_games = [game for game in games if 'f1' in game.lower()]
    print("\nF1 related games found:")
    print(f1_games)

if __name__ == "__main__":
    main() 