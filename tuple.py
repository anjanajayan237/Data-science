my_tuple = (101, "Alice", 95.5, True, [10, 20], ("Python", "Tuple"))

print("Original Tuple:")
print(my_tuple)


print("\nDisplaying Tuple Elements:")
for item in my_tuple:
    print(item)


print("\nTuple Slicing:")
print("First 3 elements:", my_tuple[:3])
print("Elements from index 2 to 4:", my_tuple[2:5])
print("Last 2 elements:", my_tuple[-2:])
print("Every second element:", my_tuple[::2])


packed_tuple = (1, "Python", 3.14, False)
print("\nPacked Tuple:")
print(packed_tuple)

num, language, value, status = packed_tuple

print("\nUnpacked Values:")
print("Number:", num)
print("Language:", language)
print("Value:", value)
print("Status:", status)