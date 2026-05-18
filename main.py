import os
import subprocess

def main():
    """Main entry point to run the Streamlit app."""
    print("Starting the Information Retrieval System...")
    
    # Path to the streamlit app
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    
    # Run streamlit
    try:
        subprocess.run(['streamlit', 'run', app_path])
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
