def repeater(times):
    def custom_deco(func):
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                result = func(*args, **kwargs)
                print('function started.')
                print('function completed.')
                results.append(result)
            return results
        return wrapper
    return custom_deco

@repeater(times=3)
def add_numbers(params):
    return sum(list(params))


if __name__ == "__main__":
    result = add_numbers([1,2,3])
    print(result)