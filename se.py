import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[10,20,15,25,30]
sns.lineplot(x=x,y=y,marker="o")
plt.title("Line plot")
plt.show()