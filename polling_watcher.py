# polling_watcher.py
import os
import time
import hashlib
import subprocess

FILE_TO_WATCH = 'knowledge_base_bilingual.json'
COMMAND_TO_RUN = ['python', 'ingest_data.py']
POLL_INTERVAL = 5  # Check for changes every 5 seconds

last_hash = ""

def get_file_hash(filename):
    """Calculates the MD5 hash of a file's content."""
    if not os.path.exists(filename):
        return None
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

print(f"Polling watcher started. Watching {FILE_TO_WATCH} for content changes...")
# Run once on startup
try:
    print("Running initial ingestion...")
    subprocess.run(COMMAND_TO_RUN, check=True)
    last_hash = get_file_hash(FILE_TO_WATCH)
    print(f"Initial ingestion complete. Current hash: {last_hash}")
except Exception as e:
    print(f"Initial ingestion failed: {e}")


while True:
    time.sleep(POLL_INTERVAL)
    current_hash = get_file_hash(FILE_TO_WATCH)

    if current_hash is not None and current_hash != last_hash:
        print(f"Content change detected. Old hash: {last_hash}, New hash: {current_hash}")
        try:
            print(f"Running command: {' '.join(COMMAND_TO_RUN)}")
            subprocess.run(COMMAND_TO_RUN, check=True)
            last_hash = current_hash
            print("Ingestion complete.")
        except Exception as e:
            print(f"Ingestion script failed: {e}")