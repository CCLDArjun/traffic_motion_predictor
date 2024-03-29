import sys
sys.path.append("..")
import os
import concurrent.futures
import traceback

import torch
from dataset import SimpleCNNDataset
import time

d = SimpleCNNDataset("../data/sets/v1.0-trainval", size="full", cache_size=0)

def process_chunk(start, end):
    try:
        start_time = time.time()
        print("starting", start, end)
        chunk_folder = "chunks/"

        os.makedirs(chunk_folder, exist_ok=True)
        
        if end > len(d):
            end = len(d)

        sample = d[start]
        torch.save(sample, chunk_folder + f"{start}-{end}.pt")
        print(sample[0].mean())

        end_time = time.time()
        print("Saved", start, end, "took", end_time - start_time, "seconds")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    num_chunks = 32185
    length_of_dataset = 33000
    chunk_size = 1
    
    # Process each chunk in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            futures.append(executor.submit(process_chunk, start, end))
        
        # Wait for all processes to finish
        concurrent.futures.wait(futures)
        for f in futures:
            print(f.result())

