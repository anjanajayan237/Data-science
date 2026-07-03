def stud_grade(percentage):
    if percentage >= 90:
        return "A"
    elif percentage  >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    elif percentage >= 50:
        return "E"
    else:
        return "F"

m1=float(input("Enter your subject1:"))
m2=float(input("Enter your subject2:"))
m3=float(input("Enter your subject3:"))
m4=float(input("Enter your subject4:"))
m5=float(input("Enter your subject5:"))
total=(m1+m2+m3+m4+m5)

percentage = (total / 500) * 100
grade=stud_grade(percentage)

print("Marks Total:",total)

print("Marks Total:",total)
print("marks percentage:",percentage,"%")
print("Grade:",grade)



