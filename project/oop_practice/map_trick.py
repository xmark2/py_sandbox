def square(n):
    return n**2

# persons = [["Han", "Solo"], ["Obi-Wan", "Kenobi"], ["Darth", "Vader"]]
#
# first, lasts = zip(*persons)

if __name__=="__main__":
    # print(first)
    # print(lasts)

    # print(list(zip(first, lasts)))
    List = [1,2,3,4,5]
    squares = list(map(square, List))
    print(squares)