import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import tkinter as tk
from os import path

#~~~~ To check the screen resolution ~~~~#
root = tk.Tk()
width_px = root.winfo_screenwidth()
height_px = root.winfo_screenheight()
width_mm = root.winfo_screenmmwidth()
height_mm = root.winfo_screenmmheight()
# 2.54 cm = in
width_in = width_mm / 25.4
height_in = height_mm / 25.4
width_dpi = width_px/width_in
height_dpi = height_px/height_in

print('Width: %i px, Height: %i px' % (width_px, height_px))
print('Width: %i mm, Height: %i mm' % (width_mm, height_mm))
print('Width: %f in, Height: %f in' % (width_in, height_in))
print('Width: %f dpi, Height: %f dpi' % (width_dpi, height_dpi))
#~~~~ ~~~~#
#~~~ To plot individual days and the average of all the days whoes data os available ~~~#
array=np.zeros([0])
count=0
datasum=np.zeros([2160])
h=int(input("Note: If you dont have the data for a paricular day the you will see an average histogram\nEnter the Day# to see the variation of intensities:"))
for f in range(1,367):
    array=np.zeros([0])
    file=path.exists("C:/Users/HP/Downloads/VSP/SnSe/Input Files/Day%03d.csv"%f)
    #print(file)
    if file == True:
        data=np.genfromtxt("C:/Users/HP/Downloads/VSP/SnSe/Input Files/Day%03d.csv"%f,delimiter=',')
        count=count+1
        #print(count)
    else:
        data=np.zeros([360])
    for q in range(len(data)):
        for j in range(2):
            if j==0:
                arr0 = [data[q] for i in range(3)]
            else:
                arr1 = [data[q] for i in range(3)]
        array1=np.append(arr0,arr1)
        array=np.append(array,array1)
    if f==h:
        arrayh=array
        print(arrayh)
    else:
        array
    datasum=datasum+array
    #print(f,len(array))
    #print(array)
    iMat = array.reshape(720,3)
    plt.imshow(iMat,'gray_r',extent=((3*f-2),(3*f),0,720))
datasum=datasum/count #Average of all the given input days
datasum1=datasum  #
iMat1 = datasum.reshape((720,3))
plt.imshow(iMat1,'gray_r',extent=((3*370-2),(3*372),0,720))
plt.xlim(0,1280) #For now the resolution of the screen is kept as 1280*720
plt.ylim(0,720)

#~~~~  y-axis label  ~~~~#
y = np.arange(360,-0.5,-0.5)
ny = y.shape[0]
noy_labels = 10 # how many labels to see on axis x
step_y = int(ny / (noy_labels - 1)) # step between consecutive labels
y_positions = np.arange(0,ny,step_y) # pixel count at label position
y_labels = y[0::step_y] # labels visible
plt.yticks(y_positions,y_labels)
plt.ylabel('Degree')
ver_line=np.zeros([720])
#~~~~  x-axis label  ~~~~#
x0 = np.arange(0,366,0.3333)
x = np.array([round(x0[p]) for p in range(len(x0))])
nx = x.shape[0]
nox_labels = 10 # how many labels to see on axis x
step_x = int((nx+1) / nox_labels)+1 # step between consecutive labels
x_positions = np.arange(0,nx+1,step_x) # pixel count at label position
x_labels = x[0::step_x] # labels visible
plt.xticks(x_positions,x_labels)
plt.xlabel('Sidereal days')
ver_line=np.zeros([720])

#~~~~ Plot for a particular day ~~~~#
#Description: This is the plot in 160*720 pixels divided into two parts.
#First 80*720 pixels showing the -ve variation of a particular day and the next 80*720 pixels showing the +variation.
arraydummy=arrayh-datasum1
for g in range(360):
    a=arrayh[6*g]
    b=datasum1[6*g]
    #c=a-b
    c=arraydummy[6*g]
    new_array=np.zeros([80])
    if c>=0:
        dimen=int(c*80)
        for o in range(dimen):
            new_array[o]=1
        x1=(3*373)+81 #Start pixel along x
        x2=x1+80 #End pixel along x
        y1=720-2*g-2 #Start pixel along y
        y2=y1+1
        new_arrayim=new_array.reshape(1,80)
        plt.imshow(new_arrayim,'gray_r',extent=(x1,x2,y1,y2))
    else:
        dimen=-int(c*80)
        for o in range(80):
            if o<=(80-dimen):
                new_array[o]=0
            else:
                new_array[o]=1
        x1=(3*373)+1
        x2=x1+80
        y1=720-2*g-2
        y2=y1+1
        new_arrayim=new_array.reshape(1,80)
        plt.imshow(new_arrayim,'gray_r',extent=(x1,x2,y1,y2))
plt.show()
