import glob
import os

def count_lines_of_code():
    total_lines = 0
    files = glob.glob('app/**/*.py', recursive=True)
    
    print(f"Found {len(files)} Python files")
    print("-" * 50)
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = len(f.readlines())
                print(f"{lines:4d} lines in {file_path}")
                total_lines += lines
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print("-" * 50)
    print(f"Total lines of code: {total_lines}")

if __name__ == "__main__":
    count_lines_of_code()