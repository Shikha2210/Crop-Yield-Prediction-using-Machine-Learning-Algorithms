from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import pickle
from sklearn.preprocessing import StandardScaler

main = tkinter.Tk()
main.title("Crop Yield Prediction using RNN, Feedforward and LSTM Neural Network")
main.geometry("1000x650")

global filename
global rnn_acc,lstm_acc, ff_acc
global classifier
global X, Y, Y1
global rainfall_dataset
global crop_dataset
global le
scalerX = StandardScaler()

global weight_for_0
global weight_for_1

def upload():
    global filename
    global rainfall_dataset
    global crop_dataset
    global le
    filename = filedialog.askdirectory(initialdir = ".")
    rainfall_dataset = pd.read_csv('dataset/district wise rainfall normal.csv')
    crop_dataset = pd.read_csv('dataset/Agriculture In India.csv')
    crop_dataset.fillna(0, inplace = True)
    crop_dataset['Production'] = crop_dataset['Production'].astype(np.int64)
    print(crop_dataset.dtypes)
    print(crop_dataset['Production'])
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    text.insert(END,str(crop_dataset.head))

        

def preprocess():
    global weight_for_0
    global weight_for_1
    global crop_dataset
    global le
    global X, Y
    text.delete('1.0', END)
    le = LabelEncoder()
    crop_dataset['State_Name'] = pd.Series(le.fit_transform(crop_dataset['State_Name']))
    crop_dataset['District_Name'] = pd.Series(le.fit_transform(crop_dataset['District_Name']))
    crop_dataset['Season'] = pd.Series(le.fit_transform(crop_dataset['Season']))
    crop_dataset['Crop'] = pd.Series(le.fit_transform(crop_dataset['Crop']))
    crop_datasets = crop_dataset.values
    cols = crop_datasets.shape[1]-1
    X = crop_datasets[:,0:cols]
    Y = crop_datasets[:,cols]
    Y = Y.astype('uint8')
    avg = np.average(Y)
    #avg = avg / 60
    Y1 = []
    for i in range(len(Y)):
        if Y[i] >= avg:
            Y1.append(1)
        else:
            Y1.append(0)
    Y = np.asarray(Y1)
    Y = Y.astype('uint8')
    a,b = np.unique(Y, return_counts=True)
    print(str(a)+" "+str(b))
    Y = to_categorical(Y)
    Y = Y.astype('uint8')
    counts = np.bincount(Y[:, 0])
    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]
    print(X.shape)
    print(Y.shape)
    scalerX.fit(X)
    X = scalerX.transform(X)
    text.insert(END,str(X))

def runRNN():
    global rnn_acc
    global X, Y
    global classifier
    text.delete('1.0', END)
    global rnn_acc
    global weight_for_0
    global weight_for_1
    if os.path.exists('model/rnnmodel.json'):
        with open('model/rnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/rnnmodel_weights.h5")
        print(classifier.summary())
        f = open('model/rnnhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        accuracy = data[1] * 100
        rnn_acc = accuracy
        text.insert(END,'RNN Prediction Accuracy : '+str(accuracy)+"\n\n")
    else:
        class_weight = {0: weight_for_0, 1: weight_for_1}
        rnn = Sequential() #creating RNN model object
        rnn.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_initializer = "uniform")) #defining one layer with 256 filters to filter dataset
        rnn.add(Dense(128, activation='relu', kernel_initializer = "uniform"))#defining another layer to filter dataset with 128 layers
        rnn.add(Dense(Y.shape[1], activation='softmax',kernel_initializer = "uniform")) #after building model need to predict two classes such as normal or Dyslipidemia disease
        rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #while filtering and training dataset need to display accuracy 
        print(rnn.summary()) #display rnn details
        rnn_acc = rnn.fit(X, Y, epochs=2, batch_size=64,class_weight=class_weight) #start building RNN model
        values = rnn_acc.history #save each epoch accuracy and loss
        values = values['accuracy']
        acc = values[1] * 100
        rnn_acc = acc;
        f = open('model/rnnhistory.pckl', 'wb')
        pickle.dump(values, f)
        f.close()
        text.insert(END,'RNN Prediction Accuracy : '+str(acc)+"\n\n")
        classifier = rnn
        classifier.save_weights('model/rnnmodel_weights.h5')
        model_json = classifier.to_json()
        with open("model/rnnmodel.json", "w") as json_file:
            json_file.write(model_json)

def runLSTM():
    global lstm_acc
    if os.path.exists('model/lstmmodel.json'):
        with open('model/lstmmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier1 = model_from_json(loaded_model_json)
        classifier1.load_weights("model/lstmmodel_weights.h5")
        print(classifier1.summary())
        f = open('model/lstmhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        accuracy = data[1] * 100
        lstm_acc = accuracy
        text.insert(END,'LSTM Prediction Accuracy : '+str(accuracy)+"\n\n")
    else:
        XX = X.reshape((X.shape[0], X.shape[1], 1)) 
        model = Sequential() #creating LSTM model object
        model.add(keras.layers.LSTM(512,input_shape=(X.shape[1], 1))) #defining LSTM layer in sequential object
        model.add(Dropout(0.5)) #removing irrelevant dataset features
        model.add(Dense(256, activation='relu'))#create another layer
        model.add(Dense(Y.shape[1], activation='softmax'))#predict two values as normal or Dyslipidemia disease
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#calculate accuracy
        print(model.summary())
        lstm_acc = model.fit(XX, Y, epochs=2, batch_size=64) #start training model
        values = lstm_acc.history
        values = values['accuracy']
        acc = values[1] * 100
        lstm_acc = acc
        f = open('model/lstmhistory.pckl', 'wb')
        pickle.dump(values, f)
        f.close()
        text.insert(END,'LSTM Prediction Accuracy : '+str(acc)+"\n\n")
        classifier1 = model
        classifier1.save_weights('model/lstmmodel_weights.h5')
        model_json = classifier1.to_json()
        with open("model/lstmmodel.json", "w") as json_file:
            json_file.write(model_json)
        

def runFF():
    global ff_acc
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    lstm_acc = model.fit(X, Y, epochs=2, batch_size=64) #start training model
    values = lstm_acc.history
    values = values['accuracy']
    ff_acc = values[1] * 100
    text.insert(END,'Feed Forward Neural Network Prediction Accuracy : '+str(ff_acc)+"\n\n")
    

def predict():
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="dataset")
    if not file:
        text.insert(END, "No file selected. Please pick dataset/testData.csv")
        return
    test = pd.read_csv(file)
    test['State_Name'] = pd.Series(le.fit_transform(test['State_Name']))
    test['District_Name'] = pd.Series(le.fit_transform(test['District_Name']))
    test['Season'] = pd.Series(le.fit_transform(test['Season']))
    test['Crop'] = pd.Series(le.fit_transform(test['Crop']))
    test = test.values
    cols = test.shape[1]
    test = test[:,0:cols]
    test = scalerX.fit_transform(test)
    #test = test.reshape((test.shape[0], test.shape[1], 1)) 
    print(test.shape)
    #test = test[:,0:test.shape[1]] 
    y_pred = classifier.predict(test)
    for i in range(len(test)):
        predict = np.argmax(y_pred[i])
        print(str(predict))
        if predict == 0:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Predicted Crop Yield will be LESS')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Predicted Crop Yield will be HIGH')+"\n\n")
    
def graph():
    global rnn_acc,lstm_acc
    bars = ['RNN Accuracy','LSTM Accuracy','Feed Forward Accuracy']
    height = [rnn_acc,lstm_acc, ff_acc]
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def topGraph():
    global rainfall_dataset
    global crop_dataset
    rainfall_dataset = pd.read_csv('dataset/district wise rainfall normal.csv')
        
    rainfall = rainfall_dataset.groupby(['STATE_UT_NAME'])['ANNUAL'].agg(['sum'])
    rainfall = rainfall.sort_values("sum", ascending=False).reset_index()
    rainfall = rainfall.loc[0:5]
    print(type(rainfall))
    rainfall = rainfall.values
    x1 = []
    y1 = []
    for i in range(len(rainfall)):
        x1.append(str(rainfall[i,0]))
        y1.append(rainfall[i,1])
    
    rice = pd.read_csv('dataset/Agriculture In India.csv')
    rice.fillna(0, inplace = True)
    rice['Production'] = rice['Production'].astype(np.int64)
    rice = rice.groupby(['State_Name','Crop'])['Production'].agg(['sum'])
    rice = rice.sort_values("sum", ascending=False).reset_index()
    x2 = []
    y2 = []
    rice = rice.values
    for i in range(len(rice)):
        if str(rice[i,1]) == 'Rice':
            x2.append(str(rice[i,0]))
            y2.append(rice[i,2])
            if len(x2) > 5:
                break;
    x3 = []
    y3 = []
    for i in range(len(rice)):
        if str(rice[i,1]) == 'Coconut':
            x3.append(str(rice[i,0]))
            y3.append(rice[i,2])
            if len(x3) > 5:
                break;
    x4 = []
    y4 = []
    for i in range(len(rice)):
        if str(rice[i,1]) == 'Sugarcane':
            x4.append(str(rice[i,0]))
            y4.append(rice[i,2])
            if len(x4) > 5:
                break;

    x5 = []
    y5 = []
    for i in range(len(rice)):
        x5.append(str(rice[i,0]))
        y5.append(rice[i,2])
        if len(x5) > 5:
            break;

    fig, ax = plt.subplots(5)
    fig.suptitle('Top 6 State Rainfall & Crop Yield')
    ax[0].plot(x1,y1.copy())
    ax[0].set_title("State Vs Rainfall")
    ax[1].plot(x2,y2.copy())
    ax[1].set_title("Top 6 State Vs Rice Crop Yield")
    ax[2].plot(x3,y3.copy())
    ax[2].set_title("Top 6 State Vs Coconut Crop Yield")
    ax[3].plot(x4,y4.copy())
    ax[3].set_title("Top 6 State Vs Sugarcane Crop Yield")
    ax[4].plot(x5,y5.copy())
    ax[4].set_title("Top 6 State Vs Any Crop Yield")
    plt.show()        

    

# ============================================================
# 10. USER INTERFACE
# ============================================================

BG        = "#eef1ee"   # window background
SIDEBAR   = "#1d3b2a"   # deep green sidebar
SIDE_DARK = "#162e21"   # sidebar section headings
ACCENT    = "#2f7d4f"   # primary button
ACCENT_HI = "#3b9c62"   # primary button hover
PANEL     = "#ffffff"   # console panel
CONSOLE   = "#111d16"   # console background
CONSOLE_FG = "#d9e9df"  # console text
INK       = "#16211a"   # dark text
MUTED     = "#7b8a80"   # secondary text

F_TITLE  = ("Segoe UI Semibold", 16)
F_SUB    = ("Segoe UI", 9)
F_SECTION = ("Segoe UI Semibold", 8)
F_BTN    = ("Segoe UI", 10)
F_PANEL  = ("Segoe UI Semibold", 10)
F_MONO   = ("Consolas", 10)
F_STATUS = ("Segoe UI", 9)

main.title("Crop Yield Prediction System")
main.minsize(880, 560)
try:
    # match the OS scaling factor so the layout fits high-DPI laptop screens
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass
try:
    main.state("zoomed")          # open maximised - fits any screen size
except Exception:
    main.geometry("1100x700")
main.configure(bg=BG)


def set_status(message, tone="idle"):
    """Update the bottom status bar. tone: idle | busy | ok | err"""
    dots = {"idle": MUTED, "busy": "#c9922e", "ok": ACCENT, "err": "#c0392b"}
    status_dot.config(fg=dots.get(tone, MUTED))
    status_label.config(text=message)


def log(message=""):
    """Append a line to the output console."""
    text.insert(END, message + "\n")
    text.see(END)


def step(func, label):
    """Wrap a pipeline function with status updates and error reporting."""
    def handler():
        set_status("Running — " + label, "busy")
        main.config(cursor="watch")
        main.update_idletasks()
        try:
            func()
            set_status("Completed — " + label, "ok")
        except Exception as exc:
            set_status("Failed — " + label, "err")
            log("")
            log("ERROR in " + label + ": " + type(exc).__name__ + " — " + str(exc))
        finally:
            main.config(cursor="")
    return handler


def clear_output():
    text.delete("1.0", END)
    set_status("Output cleared", "idle")


# ---------- header ----------
header = Frame(main, bg=SIDEBAR, height=78)
header.pack(side=TOP, fill=X)
header.pack_propagate(False)

Label(header, text="Crop Yield Prediction System", bg=SIDEBAR, fg="#ffffff",
      font=F_TITLE).pack(side=LEFT, padx=24, pady=(16, 0), anchor="w")
Label(header, text="RNN  ·  LSTM  ·  Feedforward Neural Networks",
      bg=SIDEBAR, fg="#9dc4ad", font=F_SUB).pack(side=LEFT, padx=(0, 0), pady=(22, 0))

# ---------- body ----------
body = Frame(main, bg=BG)
body.pack(side=TOP, fill=BOTH, expand=True)

sidebar = Frame(body, bg=SIDEBAR, width=274)
sidebar.pack(side=LEFT, fill=Y)
sidebar.pack_propagate(False)


def section(parent, caption):
    holder = Frame(parent, bg=SIDE_DARK)
    holder.pack(fill=X, pady=(14, 0))
    Label(holder, text=caption.upper(), bg=SIDE_DARK, fg="#8fb69f",
          font=F_SECTION, anchor="w").pack(fill=X, padx=20, pady=6)


def side_button(parent, caption, command):
    btn = Button(parent, text=caption, command=command, font=F_BTN,
                 bg=ACCENT, fg="#ffffff", activebackground=ACCENT_HI,
                 activeforeground="#ffffff", relief=FLAT, bd=0,
                 cursor="hand2", anchor="w", padx=16, pady=9,
                 highlightthickness=0)
    btn.pack(fill=X, padx=16, pady=4)
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT_HI))
    btn.bind("<Leave>", lambda e: btn.config(bg=ACCENT))
    return btn


section(sidebar, "1 · Dataset")
side_button(sidebar, "Upload Agriculture Dataset", step(upload, "Upload dataset"))
side_button(sidebar, "Preprocess Dataset", step(preprocess, "Preprocess dataset"))

section(sidebar, "2 · Neural Networks")
side_button(sidebar, "Run RNN Algorithm", step(runRNN, "RNN"))
side_button(sidebar, "Run LSTM Algorithm", step(runLSTM, "LSTM"))
side_button(sidebar, "Run Feedforward Network", step(runFF, "Feedforward network"))

section(sidebar, "3 · Results")
side_button(sidebar, "Accuracy Comparison Graph", step(graph, "Accuracy comparison"))
side_button(sidebar, "Predict Crop Yield (Test Data)", step(predict, "Prediction"))
side_button(sidebar, "Top 6 Crop Yield Graph", step(topGraph, "Top 6 crop yield"))

Label(sidebar, text="Run the steps in order — each one\ndepends on the step above it.",
      bg=SIDEBAR, fg="#6f9880", font=F_SUB, justify=LEFT,
      anchor="w").pack(side=BOTTOM, fill=X, padx=20, pady=16)

# ---------- console ----------
content = Frame(body, bg=BG)
content.pack(side=LEFT, fill=BOTH, expand=True, padx=18, pady=18)

bar = Frame(content, bg=BG)
bar.pack(fill=X, pady=(0, 8))
Label(bar, text="Output", bg=BG, fg=INK, font=F_PANEL).pack(side=LEFT)
clear_btn = Button(bar, text="Clear", command=clear_output, font=F_SUB,
                   bg=BG, fg=MUTED, activebackground=BG, activeforeground=INK,
                   relief=FLAT, bd=0, cursor="hand2", padx=10)
clear_btn.pack(side=RIGHT)

panel = Frame(content, bg=CONSOLE, highlightthickness=1,
              highlightbackground="#cfd8d2")
panel.pack(fill=BOTH, expand=True)

scroll = Scrollbar(panel)
scroll.pack(side=RIGHT, fill=Y)

text = Text(panel, bg=CONSOLE, fg=CONSOLE_FG, font=F_MONO, wrap=NONE,
            relief=FLAT, bd=0, padx=14, pady=12, insertbackground=CONSOLE_FG,
            yscrollcommand=scroll.set)
text.pack(side=LEFT, fill=BOTH, expand=True)
scroll.config(command=text.yview)

hscroll = Scrollbar(content, orient=HORIZONTAL, command=text.xview)
hscroll.pack(fill=X)
text.config(xscrollcommand=hscroll.set)

# ---------- status bar ----------
statusbar = Frame(main, bg="#dfe5e0", height=30)
statusbar.pack(side=BOTTOM, fill=X)
statusbar.pack_propagate(False)

status_dot = Label(statusbar, text="●", bg="#dfe5e0", fg=MUTED,
                   font=("Segoe UI", 10))
status_dot.pack(side=LEFT, padx=(16, 6))
status_label = Label(statusbar, text="Ready — start with 'Upload Agriculture Dataset'",
                     bg="#dfe5e0", fg=INK, font=F_STATUS)
status_label.pack(side=LEFT)
Label(statusbar, text="TensorFlow · Keras · scikit-learn", bg="#dfe5e0",
      fg=MUTED, font=F_STATUS).pack(side=RIGHT, padx=16)

log("Crop Yield Prediction System")
log("=" * 74)
log("")
log("Pipeline:  dataset  ->  preprocessing  ->  neural networks  ->  prediction")
log("Use the panel on the left, working top to bottom.")
log("")

main.mainloop()
