import os
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def extract_with_7z(archive_path, output_dir):
    # Try finding 7z in common locations or PATH
    locations = [
        "7z",
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe"
    ]
    for loc in locations:
        try:
            print(f"Trying to extract {archive_path} using {loc}...")
            # x: Extract with full paths, -y: Assume Yes on all queries, -o: Output directory
            cmd = [loc, "x", archive_path, f"-o{output_dir}", "-y"]
            if run_command(cmd):
                print(f"Successfully extracted {archive_path} using {loc}")
                return True
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Failed with {loc}: {e}")
    return False

def extract_with_py7zr(archive_path, output_dir):
    try:
        import py7zr
    except ImportError:
        print("py7zr is not installed. Installing it via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "py7zr"])
        import py7zr
    
    print(f"Extracting {archive_path} using py7zr...")
    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        archive.extractall(path=output_dir)
    print(f"Successfully extracted {archive_path} using py7zr")
    return True

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    archives = ["train.json.7z", "test.json.7z", "sample_submission.csv.7z"]
    
    for archive in archives:
        archive_path = os.path.join(workspace, archive)
        if not os.path.exists(archive_path):
            print(f"Archive not found: {archive_path}")
            continue
            
        # Try 7z first (much faster), fall back to py7zr
        if not extract_with_7z(archive_path, workspace):
            print("7-Zip extraction failed or 7-Zip not found. Falling back to py7zr...")
            extract_with_py7zr(archive_path, workspace)

if __name__ == "__main__":
    main()
