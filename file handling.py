with open("data.txt", "r") as f:
    print(f.read())
with open("output.txt", "w") as f:
    f.write("Hello, World!")
with open("data.txt", "a") as f:
    f.write("\nAppended line.")
with open("data.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        print(line.strip())
import os
if os.path.exists("data.txt"):
    os.remove("data.txt")
with open("data.txt", "w") as f:
    f.write("New file created.")
with open("data.txt", "r") as f:
    content = f.read()
    print(content)

    