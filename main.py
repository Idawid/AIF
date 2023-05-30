import os
# import cv2
import tkinter, tkinter.messagebox
import customtkinter
from PIL import ImageTk, Image
from tkinter import filedialog as fd

# from validation import validate_video_file, validate_photo_file, validate_directory

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("800x400")
root.minsize(800, 400)
root.maxsize(800, 400)
root.title("AIF project")
# root.iconbitmap("hand_sign.ico")

image1 = Image.open("demo.JPG")
image1 = image1.resize((500, 250), Image.LANCZOS)
test = ImageTk.PhotoImage(image1)


def live_button_function():
    # link to recognitionAllLetters.py
    print("recognitionAllLetters should run")


def video_button_function():
    print("recognitionAllLetters should run")


def photo_button_function():
    print("recognitionAllLetters should run")


def info_button_function():
    tkinter.messagebox.showinfo("Authors", "Course: Artificial Intelligence Fundamentals 22/23\n\n"
                                           "Authors: Dawid Mączka\n\t"
                                           "Nikodem Olszowy\n\tMateusz Sudejko\n\t"
                                           "Maciej Saju Sajecki")


frame = customtkinter.CTkFrame(master=root,
                               width=750,
                               height=350,
                               corner_radius=10)
frame.pack(padx=20, pady=20)

label = customtkinter.CTkLabel(master=frame, text="Stock Price Prediction", font=("Arial", 32))
label.place(relx=0.5, rely=0.05, anchor=tkinter.N)

label = customtkinter.CTkLabel(master=frame, text="Current Stock: TSLA", font=("Arial", 15))
label.place(relx=0.84, rely=0.87, anchor=tkinter.N)

live_button = customtkinter.CTkButton(master=frame, height=38, font=("Arial", 18), text="Refresh",
                                      command=live_button_function)
live_button.place(relx=0.85, rely=0.35, anchor=tkinter.CENTER)

video_button = customtkinter.CTkButton(master=frame, height=38, font=("Arial", 18), text="Change Stock",
                                       command=video_button_function)
video_button.place(relx=0.85, rely=0.55, anchor=tkinter.CENTER)

photo_button = customtkinter.CTkButton(master=frame, height=38, font=("Arial", 18), text="Sample Text",
                                       command=photo_button_function)
photo_button.place(relx=0.85, rely=0.75, anchor=tkinter.CENTER)

label1 = tkinter.Label(image=test)
label1.image = test
label1.place(x=40, y=100)

button = customtkinter.CTkButton(master=frame,
                                 width=20,
                                 height=20,
                                 border_width=1,
                                 corner_radius=8,
                                 text="i",
                                 fg_color="transparent",
                                 border_color="white",
                                 hover_color="gray",
                                 command=info_button_function)
button.place(relx=0.97, rely=0.93, anchor=tkinter.CENTER)

view = tkinter.StringVar(master=frame, value="off")

# switch_1 = customtkinter.CTkSwitch(master=frame, text="view non-live results",
#                                    variable=view, onvalue="on", offvalue="off")
# switch_1.place(relx=0.25, rely=0.91, anchor=tkinter.CENTER)

root.mainloop()
