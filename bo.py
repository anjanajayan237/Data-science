from bokeh.plotting import figure, show
from matplotlib import axis
x=[1,2,3,4,5]
y=[10,20,30,40,50]
p=figure(title="Line plot",x_axis_label="x",y_axis_label="y")
p.line(x,y,line_width=2)
p.circle(x,y,size=10,color="red")
show(p)