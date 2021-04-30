from tkinter import *
from tkinter import scrolledtext, font, ttk
import glob
import threading
import time
import numpy as np
import tensorflow as tf
import tensorflow_text

# Variables
global loading
global animating
global model


# Button command
def buttonCommand():
    global loading
    loading = True
    threading.Thread(target=loadingAnimation).start()
    threading.Thread(target=predict).start()


def predict():
    global loading
    global animating
    global model
    goButton["state"] = "disabled"
    textInput = titleInput.get() + " " + reviewInput.get('1.0', 'end-1c')
    textInput = re.sub(r'[^a-zA-Z\d\s:]', '', textInput)  # Remove punctuations
    textInput = re.sub(r'\s+', ' ', textInput)  # Remove multiple spaces
    textInput = re.sub(r'[ \t]+$', '', textInput)  # Remove trailing spaces
    try:
        model
    except NameError:
        modelLoad()
    prediction = model.predict(np.array([textInput]))[0]
    print(dropdown.get(), "|", textInput, "|", prediction)
    starNum = np.argmax(prediction) + 1
    loading = False
    while animating:
        continue
    stars.place(relx=0.5, rely=0.875, width=400, y=-15, x=-200)
    stars.config(text='★' * starNum, fg='#FFA41D', font=starsFont)


def loadingAnimation():
    global loading
    global animating
    dots = 0
    stars.place(relx=0.5, rely=0.875, width=400, y=-8, x=-225)
    while loading:
        animating = True
        stars.config(text='{}Loading{}'.format(' ' * dots, '.' * dots), fg='#19AFFF', font=loadingFont)
        if dots < 3:
            dots += 1
        else:
            dots = 0
        animating = False
        time.sleep(0.4)
    goButton["state"] = "normal"


# Dropdown command (activates when an option is chosen)
def dropdownCommand(*args):
    threading.Thread(target=modelLoad).start()
    threading.Thread(target=loadingAnimation).start()


# Load model
def modelLoad():
    global loading
    global model
    goButton["state"] = "disabled"
    loading = True
    selectedFile = "..\Models\\" + dropdown.get()
    model = tf.keras.models.load_model(selectedFile, compile=False)
    loading = False
    stars.config(text='')


# Dropdown update (loads list of models)
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
loadingFont = font.Font(family="Lucida Console", size=27)

# UI Elements
frame = Frame(root, bd=0, bg='white')
frame.place(relx=0, rely=0, relheight=1, relwidth=1)

title = Label(frame, text='Review Rating Suggester', font=titleFont, bg='white')
title.place(relx=0.5, rely=0.125, width=500, y=-40, x=-250)

instructions = Label(frame, text='Write a review and we\'ll suggest its rating:', font=instructionsFont, bg='white')
instructions.place(relx=0.5, rely=0.25, width=500, y=-60, x=-250)

titleInput = Entry(frame, bd=3, relief=GROOVE, font=textFont)
titleInput.place(relx=0.25, rely=0.25, height=25, relwidth=0.5, y=-30, width=-16.5)

reviewInput = scrolledtext.ScrolledText(frame, undo=True, pady=10, bd=3, relief=GROOVE, wrap=WORD, font=textFont)
reviewInput.place(relx=0.25, rely=0.25, relheight=0.5, relwidth=0.5)

buttonsFrame = Frame(root, bd=0, bg='white')
buttonsFrame.place(relx=0.25, rely=0.75, relwidth=0.5, y=10, height=25)

goButton = Button(buttonsFrame, text='Go', font=buttonFont, relief=GROOVE, command=buttonCommand)
goButton.place(relx=0, rely=0, relwidth=0.3, height=25)

options = ['']
dropdown = ttk.Combobox(buttonsFrame, values=options, state="readonly", postcommand=dropdownUpdate)
dropdownUpdate()
dropdown.current(0)
dropdown.bind("<<ComboboxSelected>>", dropdownCommand)
dropdown.place(relx=0.7, rely=0, relwidth=0.3, height=25)

stars = Label(frame, text='', font=starsFont, fg='#FFA41D', bg='white')
stars.place(relx=0.5, rely=0.875, width=400, y=-15, x=-200)

# Run application
root.mainloop()
