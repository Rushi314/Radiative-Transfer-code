import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import tkinter as tk
from os import path

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

array=np.zeros([0])
#x=1101*width_in/width_px
#y=720*height_in/height_px
count=0
datasum=np.zeros([2160])
h=int(input("Enter the Day#"))
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
datasum=datasum/count
datasum1=datasum
iMat1 = datasum.reshape((720,3))
plt.imshow(iMat1,'Reds_r',extent=((3*370-2),(3*372),0,720))
plt.xlim(0,1280) #For now the resolution of the screen is kept as 1280*720
plt.ylim(0,720)

#~~~~  y-axis label  ~~~~#
y = np.arange(360,-0.5,-0.5)
ny = y.shape[0]
noy_labels = 10 # how many labels to see on axis x
step_y = int(ny / (noy_labels - 1)) # step between consecutive labels
y_positions = np.arange(0,ny,step_y) # pixel count at label position
y_labels = y[0::step_y] # labels you want to see
plt.yticks(y_positions,y_labels)
plt.ylabel('Degree')
ver_line=np.zeros([720])
#~~~~  y-axis label  ~~~~#
x0 = np.arange(0,366,0.3333)
x = np.array([round(x0[p]) for p in range(len(x0))])
nx = x.shape[0]
nox_labels = 10 # how many labels to see on axis x
step_x = int((nx+1) / nox_labels)+1 # step between consecutive labels
x_positions = np.arange(0,nx+1,step_x) # pixel count at label position
x_labels = x[0::step_x] # labels you want to see
plt.xticks(x_positions,x_labels)
plt.xlabel('Days')
ver_line=np.zeros([720])

for w in range(720):
    ver_line[w]==1
ver_lineim=ver_line.reshape(720,1)
plt.imshow(ver_lineim,'gray_r',extent=((3*373-1),(3*373),0,720))

arraydummy=arrayh-datasum1
for g in range(360):
    a=arrayh[6*g]
    b=datasum1[6*g]
    #c=a-b
    c=arraydummy[6*g]
    if c>0:
        dimen=int(c*160) #multiply with the total number of remaining pixels you have on screen
    else:
        dimen=0
    print(a,b,c,dimen)
    new_array=np.zeros([160])
    for o in range(dimen):
        new_array[o]=1
    #new_array[dimen+1]=1
    x1=(3*373)+1#+int(a*160)
    x2=x1+160
    y1=720-2*g-2
    y2=y1+1
    new_arrayim=new_array.reshape(1,160)
    plt.imshow(new_arrayim,'gray_r',extent=(x1,x2,y1,y2))

plt.show()
