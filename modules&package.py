import math
from math import sqrt
def calculate_circle_area(radius):
    return math.pi * radius ** 2 

import random
def get_random_number(start, end):
    return random.randint(start, end)

import os
def list_files_in_directory(directory):
    return os.listdir(directory)
def read_file_contents(file_path):
    with open(file_path, 'r') as file:
        return file.read()
def write_file_contents(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)
def calculate_square_root(number):
    return sqrt(number)

import datetime
def get_current_date():
    return datetime.date.today()

