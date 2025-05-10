import os
import sys

def create_directory_structure():
    """Create the necessary directory structure for the project"""
    try:
        # Create main directories
        directories = [
            'utils',
            'data',
            'data/raw',
            'data/processed'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            else:
                print(f"Directory already exists: {directory}")
        
        # Create __init__.py files in package directories
        init_files = [
            'utils/__init__.py'
        ]
        
        for init_file in init_files:
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Package initialization\n")
                print(f"Created file: {init_file}")
            else:
                print(f"File already exists: {init_file}")
        
        print("\nDirectory structure setup complete!")
        print("Next steps:")
        print("1. Rename env_template.txt to .env and add your API keys")
        print("2. Add your university data files to the data/raw directory")
        print("3. Run 'python ingest_data.py' to process and vectorize your data")
        print("4. Run 'streamlit run app.py' to start the application")
        
    except Exception as e:
        print(f"Error creating directory structure: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(create_directory_structure()) 