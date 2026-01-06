if age >= 18:
    print("Adult")
elif age > 12:
    print("Teen")
else:
    print("Child")
# Nested conditional statements
if score >= 90:
    grade = 'A'
    if score >= 95:
        grade += '+'
elif score >= 80:
    grade = 'B'
else:
    grade = 'C'
# Ternary conditional operator
status = "Pass" if marks >= 40 else "Fail"
print(3 in my_list)  # True
print(6 not in my_list)  # True
# Identity operators
x = [1, 2, 3]
y = x
print(x is y)  # True
print(x is not y)  # False
print(x == y)  # True
print(x != y)  # False
# Bitwise operators
a = 5  # 0101 in binary
b = 3  # 0011 in binary
print(a & b)  # 1 (0001 in binary)
print(a | b)  # 7 (0111 in binary)
print(a ^ b)  # 6 (0110 in binary)
print(~a)     # -6 (inverts bits)
print(a << 1) # 10 (1010 in binary)
print(a >> 1) # 2 (0010 in binary)