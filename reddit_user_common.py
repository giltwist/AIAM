# Tkinter info from https://www.geeksforgeeks.org/python/python-tkinter-entry-widget/

import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox 

def load_dataset(red_row=100,user_row=100):
    # Load dataset
    redditCSV = pd.read_csv("./datasets/reddit-comments-bodyonly.csv") 
    userCSV = pd.read_csv("./datasets/user-comments-bodyonly.csv") 
    redditCSV.columns=['text']
    redditCSV['label']='reddit'
    userCSV.columns=['text']
    userCSV['label']='user'


    df=pd.concat([redditCSV.head(n=red_row),userCSV.head(n=user_row)],ignore_index=True)
    return df

def tk_init(self, function):
    self.window=tk.Tk()
    self.window.geometry=("800x640")
    self.window.title("Me or Not?")

    # creating a label for 
    # name using widget Label
    self.text_label = tk.Label(self.window, text = 'Say something only I would say')
    
    # creating a entry for input
    # name using widget Entry
    self.text_entry = scrolledtext.ScrolledText(self.window,wrap=tk.WORD, width=40, height=8)

    # creating a button using the widget 
    # Button that will call the submit function 
    self.sub_btn=tk.Button(self.window, text = 'Submit', command = function, state=tk.DISABLED)

    def update_btn(event):
        self.sub_btn.config(state=tk.NORMAL if len(self.text_entry.get("1.0", tk.END))>20 else tk.DISABLED)

    self.text_entry.bind('<KeyRelease>', update_btn)

    

    # placing the label and entry in
    # the required position using grid
    # method
    self.text_label.grid(row=0,column=0, columnspan=2)
    self.text_entry.grid(row=1,column=0, columnspan=2)
    self.sub_btn.grid(row=2,column=1,sticky="SE")
    self.window.mainloop()
    

def show_result(self, is_Me):
    if is_Me:
        messagebox.showinfo("Result","This was written by me")
    else:
        messagebox.showerror("Result","This was not written by me")
 
