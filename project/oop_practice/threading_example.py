import pandas as pd
import threading

# Define a function to process a DataFrame
def process_file(file_path, results, index):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    # df["processed_column"] = df["some_column"] * 2  # Example transformation
    results[index] = df
    print(f"Finished processing {file_path}")

if __name__ == "__main__":
    # File paths to process
    file_paths = ["file1.csv", "file2.csv", "file3.csv"]
    results = [None] * len(file_paths)

    # Create and start threads
    threads = []
    for i, file_path in enumerate(file_paths):
        # print(pd.read_csv(file_path).head())
        t = threading.Thread(target=process_file, args=(file_path, results, i))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Combine results into a single DataFrame
    final_df = pd.concat(results)
    print(final_df)