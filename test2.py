i = input()

split = i.split(" ")
number = []
for n in split:
    number.append(int(n))

number.sort(reverse=True)
print(number[2])