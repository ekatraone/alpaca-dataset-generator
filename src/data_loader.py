import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import read_file, chunk_text
from config import CONFIG

SUPPORTED_EXTENSIONS = ('.txt', '.pdf', '.docx', '.md', '.csv', '.json', '.jsonl')

def load_input_data(input_folder: str) -> List[str]:
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files
        if file.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    print(f"Found {len(all_files)} supported files to process")

    def process_file(file_path):
        try:
            text = read_file(file_path)
            chunks = chunk_text(
                text,
                chunk_size=CONFIG.get('chunk_size', 1500),
                overlap=CONFIG.get('chunk_overlap', 200),
            )
            print(f"  {os.path.basename(file_path)}: {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    texts = []
    with tqdm(total=len(all_files), desc="Loading input files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            futures = {executor.submit(process_file, fp): fp for fp in all_files}
            for future in as_completed(futures):
                texts.extend(future.result())
                pbar.update(1)

    print(f"Total chunks loaded: {len(texts)}")
    return texts
