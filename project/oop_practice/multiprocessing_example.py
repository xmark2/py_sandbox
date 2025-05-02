from multiprocessing import Pool, cpu_count

# Function to compute square
def compute_square(n):
    return n ** 2

if __name__ == "__main__":
    with Pool(processes=4) as pool:  # Use 4 parallel processes
        numbers = [1, 2, 3, 4, 5]
        results = pool.map(compute_square, numbers)
        print("Squares:", results)

    print(cpu_count())