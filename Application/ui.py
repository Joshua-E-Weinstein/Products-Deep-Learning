from tkinter import *
from tkinter import scrolledtext, font, ttk
import glob
import numpy as np
import tensorflow as tf


# Button command
def buttonCommand():
    textInput = reviewInput.get('1.0', 'end-1c')
    selectedFile = "..\Models\\" + dropdown.get()
    model = tf.keras.models.load_model(selectedFile)
    prediction = model.predict(np.array([textInput]))[0][0]
    print(selectedFile, " | ", textInput, " | ", prediction)
    starNum = round(prediction*5) if prediction > 0.05 else 1
    stars.config(text='★' * starNum)


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

stars = Label(frame, text='', font=('bold', 30), fg='#FFA41D', bg='white')
stars.place(relx=0.5, rely=0.875, width=200, y=-15, x=-100)

# Run application
root.mainloop()
