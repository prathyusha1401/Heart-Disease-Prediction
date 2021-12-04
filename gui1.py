import tkinter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sn

def takeInput():
    inputValues = []

    
    age1 = ((int(age.get()) - 29)  / (77-29 ))
    print(age1)
    trestbps1 = ((int(rbp.get()) - 94)/(200-94))
    chol1 = ((int (serumChol.get()) - 126)/(564-126))
    thalach1 = ((int(thalach.get()) - 71)/(202-71))
    oldpeak1 = (float(oldpeak.get())/ (6.2))
    
    inputValues.append(age1)
    inputValues.append(sex.get())
    inputValues.append(chestPain.get())
    inputValues.append(trestbps1)
    inputValues.append(chol1)
    inputValues.append(FBS.get())
    inputValues.append(ECG.get())
    inputValues.append(thalach1)
    inputValues.append(trestbps1)
    inputValues.append(oldpeak1)
    inputValues.append(slope.get())
    inputValues.append(ca.get())
    inputValues.append(thal.get()) 
    
    print(inputValues)


    print("\n") 
    final_Result = model.predict([inputValues])
    print(final_Result)
    
    substituteWindow = tkinter.Tk()
    substituteWindow.geometry('1920x1080')
    
    substituteWindow.columnconfigure(0, weight=2)
    substituteWindow.columnconfigure(1, weight=1)
    substituteWindow.columnconfigure(2, weight=2)
    substituteWindow.columnconfigure(3, weight=2)
    substituteWindow.rowconfigure(0, weight=1)
    substituteWindow.rowconfigure(1, weight=10)
    substituteWindow.rowconfigure(2, weight=10)
    substituteWindow.rowconfigure(3, weight=1)
    substituteWindow.rowconfigure(4, weight=1)
    substituteWindow.rowconfigure(5, weight=1)
    
    if final_Result[0] == 1:
        label1 = tkinter.Label(substituteWindow, text="!!!   HEART DISEASE DETECTED   !!!", font=('Impact', -35), fg='red')
        label1.grid(row=0, column=0, columnspan=6)
        label2 = tkinter.Label(substituteWindow, text="PLEASE VISIT NEAREST CARDIOLOGIST AT THE EARLIEST", font=('Impact', -20), fg='#0080ff')
        label2.grid(row=1, column=0, columnspan=6)
        label3 = tkinter.Label(substituteWindow, text = "Avoid HyperTension", font=('Times', -20))
        label3.grid(row=3, column = 0,columnspan=6)
        label4 = tkinter.Label(substituteWindow, text = "Maintain Cholestrol Rate", font=('Times', -20))
        label4.grid(row=4, column = 0,columnspan=6)
        label4 = tkinter.Label(substituteWindow, text = "Heavy Cardio Workout Should Be Avoided", font=('Times', -20))
        label4.grid(row=5, column = 0,columnspan=6)
    else: 
        label1 = tkinter.Label(substituteWindow, text="NO DETECTION OF HEART DISEASE", font=('Impact', -35) ,fg='green')
        label1.grid(row=1, column=0, columnspan=6)
        label2 = tkinter.Label(substituteWindow, text="Do not forget to exercise daily. ", font=('Impact', -20), fg='black')
        label2.grid(row=2, column=0, columnspan=6)      
        
    substituteWindow.mainloop()
        
        

heart = pd.read_csv("heart.csv")
# we have unknown values '?'
# change unrecognized value '?' into mean value through the column
min_max = MinMaxScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart[columns_to_scale ] = min_max.fit_transform(heart[columns_to_scale])
y = heart['target']
X = heart.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

model = LogisticRegression()
model.fit(X_train, y_train)

    
mainWindow = tkinter.Tk()
mainWindow.geometry('1920x1080')


mainWindow.columnconfigure(0, weight=2)
mainWindow.columnconfigure(1, weight=2)
mainWindow.columnconfigure(2, weight=2)
mainWindow.columnconfigure(3, weight=2)
mainWindow.rowconfigure(0, weight=0)
mainWindow.rowconfigure(1, weight=0)
mainWindow.rowconfigure(2, weight=1)
mainWindow.rowconfigure(3, weight=1)
mainWindow.rowconfigure(4, weight=1)
mainWindow.rowconfigure(5, weight=1)
mainWindow.rowconfigure(6, weight=1)
mainWindow.rowconfigure(7, weight=1)
mainWindow.rowconfigure(8, weight=5)



label1 = tkinter.Label(mainWindow,text="Heart Disease Prediction Model",font=("Courier",40,"bold"))
label1.grid(row=2, column=0, columnspan=6)

label2 = tkinter.Label(mainWindow, text="Enter the details carefully", font=('Arial',-20,"bold") )
label2.grid(row=3, column=0, columnspan=6)


#frame for the feature inputs
ageFrame = tkinter.LabelFrame(mainWindow, text="Age(yrs)")
ageFrame.grid(row=4, column=0)
ageFrame.config(font=("Courier", -17))
age= tkinter.Entry(ageFrame)
age.grid(row=2, column=2, sticky='nw')

sexFrame = tkinter.LabelFrame(mainWindow, text="Sex[1-M,0-F]")
sexFrame.grid(row=4, column=1)
sexFrame.config(font=("Courier", -17))
sex= tkinter.Entry(sexFrame)
sex.grid(row=2, column=2, sticky='nw')

chestPainFrame = tkinter.LabelFrame(mainWindow, text="CP (0-4)")
chestPainFrame.grid(row=4, column=2)
chestPainFrame.config(font=("Courier", -17))
chestPain= tkinter.Entry(chestPainFrame)
chestPain.grid(row=2, column=2, sticky='nw')


rbpFrame = tkinter.LabelFrame(mainWindow, text="RBP (94-200)")
rbpFrame.grid(row=4, column=3)
rbpFrame.config(font=("Courier", -17))
rbp= tkinter.Entry(rbpFrame)
rbp.grid(row=2, column=2, sticky='nw')

serumCholFrame = tkinter.LabelFrame(mainWindow, text="Serum Chol")
serumCholFrame.grid(row=5, column=0)
serumCholFrame.config(font=("Courier", -17))
serumChol = tkinter.Entry(serumCholFrame)
serumChol.grid(row=2, column=2, sticky='n')

FBSFrame = tkinter.LabelFrame(mainWindow, text="Fasting BP(0-4)")
FBSFrame.grid(row=5, column=1)
FBSFrame.config(font=("Courier", -17))
FBS= tkinter.Entry(FBSFrame)
FBS.grid(row=2, column=2, sticky='nw')

ECGFrame = tkinter.LabelFrame(mainWindow, text="ECG (0,1,2)")
ECGFrame.grid(row=5, column=2)
ECGFrame.config(font=("Courier", -17))
ECG = tkinter.Entry(ECGFrame)
ECG.grid(row=2, column=2, sticky='nw')


thalachFrame = tkinter.LabelFrame(mainWindow, text="thalach(71-202)")
thalachFrame.grid(row=5, column=3)
thalachFrame.config(font=("Courier", -17))
thalach = tkinter.Entry(thalachFrame)
thalach.grid(row=2, column=2, sticky='nw')

exangFrame = tkinter.LabelFrame(mainWindow, text="exAngina(0/1)")
exangFrame.grid(row=6, column=0)
exangFrame.config(font=("Courier", -17))
exang = tkinter.Entry(exangFrame)
exang.grid(row=2, column=2, sticky='nw')


oldpeakFrame = tkinter.LabelFrame(mainWindow, text="Old Peak(0-6.2)")
oldpeakFrame.grid(row=6, column=1)
oldpeakFrame.config(font=("Courier", -17))
oldpeak = tkinter.Entry(oldpeakFrame)
oldpeak.grid(row=2, column=2, sticky='nw')
  
slopeFrame = tkinter.LabelFrame(mainWindow, text="Slope(0,1,2)")
slopeFrame.grid(row=6, column=2)
slopeFrame.config(font=("Courier", -17))
slope = tkinter.Entry(slopeFrame)
slope.grid(row=2, column=2, sticky='nw')

caFrame = tkinter.LabelFrame(mainWindow, text="Major Vessels(0-3)")
caFrame.grid(row=6, column=3)
caFrame.config(font=("Courier", -17))
ca = tkinter.Entry(caFrame)
ca.grid(row=2, column=2, sticky='nw')


thalFrame = tkinter.LabelFrame(mainWindow, text=" THAL(0,1,2,3)")
thalFrame.grid(row=7, column=1)
thalFrame.config(font=("Courier", -17))
thal = tkinter.Entry(thalFrame)
thal.grid(row=2, column=2, sticky='nw')


analyseButton = tkinter.Button(mainWindow, text="..................ANALYZE/ PREDICT.....................", font=('Arial', -17,"bold"), bg = 'red', command=takeInput)
analyseButton.grid(row=8, column=0, columnspan=10)

mainWindow.mainloop()