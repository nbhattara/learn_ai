#list(mutable,ordered)
my_list = [1, 2, 3, 4, 5]
my_list.append(6)          # Add an element 
my_list.remove(3)         # Remove an element
print(my_list[0])         # Access first element
print(len(my_list))       # Length of the list
#tuple(immutable,ordered)
my_tuple = (1, 2, 3, 4, 5)
print(my_tuple[1])        # Access second element
print(len(my_tuple))      # Length of the tuple
#dictionary(mutable,unordered,key-value pairs)
my_dict = {'name': 'Alice', 'age': 25}
my_dict['age'] = 26       # Update value
my_dict['city'] = 'NY'    # Add new key-value pair
print(my_dict['name'])     # Access value by key
print(len(my_dict))       # Length of the dictionary
#set(mutable,unordered,unique elements)
my_set = {1, 2, 3, 4, 5}
my_set.add(6)             # Add an element
my_set.remove(2)          # Remove an element
print(3 in my_set)        # Check membership
print(len(my_set))        # Length of the set
# --- IGNORE ---
