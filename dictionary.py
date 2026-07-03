dict1={
    "name":"anju",
    "age":18,
    "course":"mca"
}
print("name:",dict1["name"])
dict1["college"]= "mangalam college"
dict1["age"]=22
del dict1["course"]
print("all key value pairs in the dictionary:")
for key,value in dict1.items():
    print(key,":",value)

