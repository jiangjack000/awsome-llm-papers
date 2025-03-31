import random

def rand5():
    return random.randint(1, 5)

def rand7():
    while True:
        # Generate a number from 1 to 25
        # 0,4 * 5 = 0, 20 + 1, 5 = 21, 25 
        num = (rand5() - 1) * 5 + rand5()
        if num <= 21:
            # Map the result to 1 to 7
            return (num - 1) % 7 + 1

# Example usage
mymap = {}
for _ in range(1000000):
    r = rand7()
    mymap[r] = mymap.get(r, 0) + 1


for k, v in mymap.items():
    print(f'{k}: {v}')