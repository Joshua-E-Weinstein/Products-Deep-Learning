from tkinter import *
from tkinter import scrolledtext, font, ttk
import glob
import threading
import time
import numpy as np
import tensorflow as tf

# Variables
global loading
global animating

# Button command
def buttonCommand():
    global loading
    loading = True
    threading.Thread(target=loadingAnimation).start()
    threading.Thread(target=predict).start()


def predict():
    global loading
    global animating
    textInput = reviewInput.get('1.0', 'end-1c')
    selectedFile = "..\Models\\" + dropdown.get()
    model = tf.keras.models.load_model(selectedFile)
    prediction = model.predict(np.array([textInput]))[0][0]
    print(selectedFile, " | ", textInput, " | ", prediction)
    starNum = round(prediction * 5) if prediction > 0.05 else 1
    loading = False
    while animating:
        continue
    stars.place(relx=0.5, rely=0.875, width=400, y=-15, x=-200)
    stars.config(text='★' * starNum, fg='#FFA41D')


def loadingAnimation():
    global loading
    global animating
    dots = 0
    stars.place(relx=0.5, rely=0.875, width=400, y=-15, x=-225)
    while loading:
        animating = True
        stars.config(text='{}Loading{}'.format(' ' * dots, '.' * dots), fg='#19AFFF')
        if dots < 3:
            dots += 1
        else:
            dots = 0
        animating = False
        time.sleep(0.4)


# Dropdown update
def dropdownUpdate(*args):
    dirtyOptions = glob.glob('..\Models\*\\')
    if dirtyOptions:
        cleanOptions = []
        for file in dirtyOptions:
            cleanOptions.append(file.lstrip('..\Models\\').rstrip('\\'))
        dropdown['values'] = cleanOptions
    else:
        dropdown['values'] = ['']


# Create window
root = Tk()

root.title('Ratings AI')
root.geometry('950x600')
root.minsize(700, 500)

# Fonts
titleFont = font.Font(family="Lucida Console", size=18, weight='bold')
instructionsFont = font.Font(family="Lucida Console", size=14)
textFont = font.Font(family="Calibri", size=12)
buttonFont = font.Font(family="Lucida Console", size=12)
starsFont = font.Font(family="Lucida Console", size=30)

# UI Elements
frame = Frame(root, bd=0, bg='white')
frame.place(relx=0, rely=0, relheight=1, relwidth=1)

title = Label(frame, text='Review Rating Suggester', font=titleFont, bg='white')
title.place(relx=0.5, rely=0.125, width=500, y=-18, x=-250)

instructions = Label(frame, text='Write a review and we\'ll suggest its rating:', font=instructionsFont, bg='white')
instructions.place(relx=0.5, rely=0.25, width=500, y=-30, x=-250)

reviewInput = scrolledtext.ScrolledText(frame, undo=True, padx=10, pady=10, bd=3, relief=GROOVE, wrap=WORD)
reviewInput.configure(font=textFont)
reviewInput.place(relx=0.25, rely=0.25, relheight=0.5, relwidth=0.5)

buttonsFrame = Frame(root, bd=0, bg='white')
buttonsFrame.place(relx=0.25, rely=0.75, relwidth=0.5, y=10, height=25)

goButton = Button(buttonsFrame, text='Go', font=buttonFont, relief=GROOVE, command=buttonCommand)
goButton.place(relx=0, rely=0, relwidth=0.3, height=25)

options = ['']
dropdown = ttk.Combobox(buttonsFrame, values=options, state="readonly", postcommand=dropdownUpdate)
dropdownUpdate()
dropdown.current(0)
dropdown.bind("<<ComboboxSelected>>", dropdownUpdate)
dropdown.place(relx=0.7, rely=0, relwidth=0.3, height=25)

stars = Label(frame, text='', font=starsFont, fg='#FFA41D', bg='white')
stars.place(relx=0.5, rely=0.875, width=400, y=-15, x=-200)

# Run application
root.mainloop()
