# -*- coding: utf-8 -*-
import dash
from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context, ctx
import dash.dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import os
from flask import Flask
import uuid
import time

#Imports fuer die Zusatzfenster
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import io
import base64
from transformers import pipeline

import sounddevice as sd
import soundfile as sf
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PyQt6 import QtWidgets
import sys
import librosa
import librosa.display
from PIL import Image, ImageTk
import webbrowser
import threading
import multiprocessing
import plotly.graph_objs as go  # >>> Aenderung
from queue import Queue  # >>> Aenderung
import plotly.graph_objs as go
import socketio

current_page = 1


'''
#Hier befinden sich alle Klassen, die ueber den Client aufgerufen werden muessen
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
'''
#ist fuer Seite 94
class MaskUnmasker:
    @staticmethod
    def unmask_sentence(text, model_name="bert-base-cased"):
        """Ersetzt das [MASK]-Token im gegebenen Text durch das wahrscheinlichste Wort."""
        fill_mask = pipeline("fill-mask", model=model_name)
        predictions = fill_mask(text)
        best_prediction = predictions[0]['sequence']  # Beste Vorhersage ausgeben
        return best_prediction

# Beispielverwendung
unmasker = MaskUnmasker()
input_text = "I feel very [MASK] today."
output_text = unmasker.unmask_sentence(input_text)

("Eingabe:", input_text)
print("Unmaskierter Satz:", output_text)


########Klassen und Methoden zum 
#Code fuer blances Fenster zum Notizen machen
class BlancFenster:
    @staticmethod
    def open_drawing_window_in_process():
        """ Starts open_drawing_window() in its own process """
        tk_process = multiprocessing.Process(target=BlancFenster._run_tkinter)
        tk_process.start()

    @staticmethod
    def _run_tkinter():
        """ Separate function for the multiprocessing process """
        root = tk.Tk()  # Create Tk main window
        BlancFenster.open_drawing_window(root)  # Call drawing window within Tk
        root.mainloop()  # mainloop must run in separate process

    @staticmethod
    def open_drawing_window(root):
        """ Opens a blank drawing window for free drawing with mouse """
        width = 600  # Width of the canvas
        height = 400  # Height of the canvas

        drawing_window = tk.Toplevel(root)  # Create drawing window in Tk
        drawing_window.title("Zeichenfenster")

        canvas = tk.Canvas(drawing_window, width=width, height=height, bg="white")
        canvas.pack()

        # Initial position for drawing
        last_position = None

        # Function to start drawing
        def start_drawing(event):
            nonlocal last_position
            last_position = (event.x, event.y)

        # Function to draw freehand as mouse moves
        def draw_freehand(event):
            nonlocal last_position
            current_position = (event.x, event.y)
            if last_position:
                canvas.create_line(last_position[0], last_position[1], current_position[0], current_position[1], fill="black", width=2)
            last_position = current_position

        # Function to end drawing
        def end_drawing(event):
            nonlocal last_position
            last_position = None

        # Bind mouse events to the canvas
        canvas.bind("<ButtonPress-1>", start_drawing)
        canvas.bind("<B1-Motion>", draw_freehand)
        canvas.bind("<ButtonRelease-1>", end_drawing)


########Klassen und Methoden zum 
#### Code fuer Zeichenfenster
class Zeichenfenster:
    @staticmethod
    def open_drawing_window_in_process():
        """ Startet open_drawing_window() in einem eigenen Prozess """
        tk_process = multiprocessing.Process(target=Zeichenfenster._run_tkinter)
        tk_process.start()

    @staticmethod
    def _run_tkinter():
        """ Separate Funktion fuer den Multiprocessing-Prozess """
        root = tk.Tk()  # Tk Hauptfenster erstellen
        Zeichenfenster.open_drawing_window(root)  # Zeichenfenster innerhalb von Tk aufrufen
        root.mainloop()  # mainloop muss im separaten Prozess laufen

    @staticmethod
    def open_drawing_window(root, x_neg=10, x_pos=10, y_neg=10, y_pos=10):
        """ Oeffnet ein Zeichenfenster mit Koordinatensystem und Pfeilzeichnung """
        scale = 40  # Pixel pro Einheit
        width = (x_neg + x_pos) * scale
        height = (y_neg + y_pos) * scale

        drawing_window = tk.Toplevel(root)  # Zeichenfenster in Tk
        drawing_window.title("Zeichenfenster mit Koordinatensystem")

        canvas = tk.Canvas(drawing_window, width=width, height=height, bg="white")
        canvas.pack()

        # Nullpunkt dynamisch setzen
        center_x = x_neg * scale
        center_y = y_pos * scale  

        # Koordinatensystem zeichnen
        canvas.create_line(center_x, 0, center_x, height, fill="black", width=2)
        canvas.create_line(0, center_y, width, center_y, fill="black", width=2)

        # X-Achse markieren
        for i in range(-x_neg, x_pos + 1):
            x_pos_pixel = center_x + (i * scale)
            canvas.create_text(x_pos_pixel, center_y + 10, text=str(i), font=("Arial", 10))
            canvas.create_line(x_pos_pixel, center_y - 5, x_pos_pixel, center_y + 5, fill="black")

        # Y-Achse markieren
        for i in range(-y_neg, y_pos + 1):
            y_pos_pixel = center_y - (i * scale)  
            canvas.create_text(center_x + 10, y_pos_pixel, text=str(i), font=("Arial", 10))
            canvas.create_line(center_x - 5, y_pos_pixel, center_x + 5, y_pos_pixel, fill="black")

        start_x, start_y = None, None
        current_arrow = None

        # Mausklick startet das Zeichnen
        def start_drawing(event):
            nonlocal start_x, start_y, current_arrow
            start_x, start_y = event.x, event.y
            current_arrow = canvas.create_line(start_x, start_y, start_x, start_y, arrow=tk.LAST, width=2, fill="red")

        # Mausbewegung aktualisiert den Pfeil
        def draw_arrow(event):
            nonlocal current_arrow
            if current_arrow:
                canvas.coords(current_arrow, start_x, start_y, event.x, event.y)

        # Maustaste loslassen beendet das Zeichnen
        def finish_drawing(event):
            nonlocal current_arrow
            current_arrow = None  

        # Mausereignisse binden
        canvas.bind("<ButtonPress-1>", start_drawing)  
        canvas.bind("<B1-Motion>", draw_arrow)        
        canvas.bind("<ButtonRelease-1>", finish_drawing)  

class CameraApp:
    @staticmethod
    def open_camera_window_in_process():
        """Startet das Kamera-Fenster in einem separaten Prozess."""
        tk_process = multiprocessing.Process(target=CameraApp._run_tkinter)
        tk_process.start()

    @staticmethod
    def _run_tkinter():
        """Initialisiert Tkinter in einem separaten Prozess."""
        root = tk.Tk()
        app = CameraApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()

    def __init__(self, root):
        self.root = root
        self.root.title("Main Window")
        self.cap = None

        open_button = tk.Button(self.root, text="Open Camera Window", command=self.open_camera_window)
        open_button.pack(pady=20)

    def start_video_stream(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to initialize the camera.")
            return False
        return True

    def update_image(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label_image.imgtk = imgtk  # Prevent garbage collection
                self.label_image.config(image=imgtk)
                self.label_image.after(10, self.update_image)

    def capture_image(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                base64_encoded_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                return img, base64_encoded_data
            else:
                messagebox.showerror("Error", "Failed to capture the image.")
                return None, None

    def classify_image(self):
        img, encoded = self.capture_image()
        if img and encoded:
            classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
            emotion = classifier(img)
            messagebox.showinfo("Result", str(emotion))
        else:
            messagebox.showerror("Error", "Failed to classify image.")

    def open_camera_window(self):
        camera_window = tk.Toplevel(self.root)
        camera_window.title("Camera Window")

        if not self.start_video_stream():
            camera_window.destroy()
            return

        self.label_image = tk.Label(camera_window)
        self.label_image.pack()

        self.update_image()

        capture_button = tk.Button(camera_window, text="Capture and classify image", command=self.classify_image)
        capture_button.pack()

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


class AudioApp:
    """ Klasse zur Audioaufnahme mit einer statischen Queue. """
    audio_queue = Queue()  # Statische Queue fuer Audiodaten
    audio_list = None ###Dort sind die Audio-Daten zum Abruf in Dash zwischengespeichert
    audio_list_length = 50
    audio_flag = False  # Flag zur Steuerung der Aufnahme
    recording_thread = None  # Referenz auf den Queue-Auslese-Thread
    stream = None  # Sounddevice-Stream

    @staticmethod
    def start_recording():
        """ Startet die Audioaufnahme und das Queue-Auslesen. """
        if AudioApp.audio_flag:
            print("Aufnahme laeuft bereits.")
            return

        with AudioApp.audio_queue.mutex:
            AudioApp.audio_queue.queue.clear()  # Alte Daten loeschen
        
        AudioApp.audio_flag = True  # Setzt die Flag auf aktiv
        print("Aufnahme gestartet.")

        # Startet den Sound-Stream fuer die Aufnahme
        AudioApp.stream = sd.InputStream(
            callback=AudioApp.audio_callback, samplerate=44100, channels=1, dtype='float32'
        )
        AudioApp.stream.start()

        # Starte den Queue-Auslese-Thread, falls er nicht laeuft
        #if AudioApp.recording_thread is None or not AudioApp.recording_thread.is_alive():
        #    AudioApp.recording_thread = threading.Thread(target=AudioApp.print_queue_data, daemon=True)
        #    AudioApp.recording_thread.start()

    @staticmethod
    def stop_recording():
        """ Stoppt die Aufnahme und das Queue-Auslesen. """
        if AudioApp.audio_flag:
            #bugfixing
            #print(AudioApp.get_queue_data_10())
            print("Audio Aufnahme wird beendet.")
            AudioApp.audio_flag = False  # Flag auf False setzen
            if AudioApp.stream is not None:
                AudioApp.stream.stop()
                AudioApp.stream.close()
                AudioApp.stream = None
            AudioApp.audio_flag = False
            # Warten, bis der Thread sich beendet hat
            if AudioApp.recording_thread is not None:
                print("[DEBUG] Warte auf das Beenden des Threads...")
                AudioApp.recording_thread.join(timeout=1)
                AudioApp.recording_thread = None
        
        print("Aufnahme gestoppt.")

    @staticmethod
    def audio_callback(indata, frames, time, status):
        """ Fuegt eingehende Audiodaten zur statischen Queue hinzu. """
        if status:
            print("Audio Status:", status)
        AudioApp.audio_queue.put(indata.copy())  # Speichert Audiodaten in der Queue
        AudioApp.audio_list = AudioApp.get_queue_data_modular()
        print(AudioApp.audio_list)

    @staticmethod
    def print_queue_data():
        """ Liest alle 50ms die Queue aus, solange die Aufnahme aktiv ist. """
        while AudioApp.audio_flag:
            if not AudioApp.audio_queue.empty():
                audio_chunk = AudioApp.audio_queue.get()
                print(f"Audio-Daten: {audio_chunk[:10]} ...")  # Zeigt einen Teil der Daten an
            else:
                print("Queue ist leer.")
            sd.sleep(50)  # 50 Millisekunden warten

    @staticmethod
    def get_queue_data_10():
        """ Holt einen Ausschnitt der Audiodaten, falls vorhanden. """
        if not AudioApp.audio_queue.empty():
            raw_data = AudioApp.audio_queue.get()[:10]
            return [x[0] for x in raw_data] 
        return None

    @staticmethod
    def get_queue_data_100():
        """ Holt einen Ausschnitt der Audiodaten, falls vorhanden. """
        if not AudioApp.audio_queue.empty():
            raw_data = AudioApp.audio_queue.get()[:1000:10]
            return [x[0] for x in raw_data] 
        return None


    @staticmethod
    def get_queue_data_modular(interval = 20):
        """ Holt einen Ausschnitt der Audiodaten, falls vorhanden. """
        if not AudioApp.audio_queue.empty():
            queue_ausschnitt = AudioApp.audio_list_length*interval
            raw_data = AudioApp.audio_queue.get()[:queue_ausschnitt:interval]
            return [x[0] if x[0]< -0.001 or x > 0.001 else 0 for x in raw_data] 
        return None


class YoutubeKommentarApp:
    @staticmethod
    def open_youtube_window_in_process():
        """Startet das YouTube-Kommentarfeld in einem separaten Prozess."""
        tk_process = multiprocessing.Process(target=YoutubeKommentarApp._run_tkinter)
        tk_process.start()

    @staticmethod
    def _run_tkinter():
        """Initialisiert Tkinter in einem separaten Prozess."""
        root = tk.Tk()
        app = YoutubeKommentarApp(root)
        root.mainloop()

    def __init__(self, master):
        self.master = master
        self.master.title("YouTube Kommentar-Analyse")
        self.master.geometry("600x500")

        self.info_label = tk.Label(
            master,
            text="Wir werden im Folgenden die Emotionen eines YouTube-Kommentars auswerten.\n"
                 "Oeffne YouTube mit dem Button, suche ein Video, kopiere einen Kommentar\n"
                 "und fuege ihn hier ein.",
            font=("Arial", 12),
            fg="black"
        )
        self.info_label.pack(pady=10)

        self.youtube_button = tk.Button(
            master,
            text="YouTube-Suche oeffnen",
            command=self.open_youtube,
            font=("Arial", 12)
        )
        self.youtube_button.pack(pady=10)

        self.text_label = tk.Label(master, text="Hier den YouTube-Kommentar einfuegen:", font=("Arial", 12))
        self.text_label.pack(pady=10)

        self.text_input = tk.Text(master, height=2, width=50, font=("Arial", 12))
        self.text_input.pack(pady=10)

        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)

        self.analyze_button = tk.Button(button_frame, text="Auswerten", command=self.analyze_text, font=("Arial", 12))
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(button_frame, text="Leeren", command=self.clear_text, font=("Arial", 12))
        self.clear_button.pack(side=tk.RIGHT, padx=5)

        self.result_label = tk.Label(master, text="Ergebnis:", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

        self.result_output = tk.Text(master, height=5, width=60, font=("Arial", 12))
        self.result_output.pack(pady=10)

    def analyze_text(self):
        user_text = self.text_input.get("1.0", tk.END).strip()
        if not user_text:
            messagebox.showwarning("Warnung", "Bitte Text eingeben!")
            return

        classifier2 = pipeline("text-classification", model="MilaNLProc/xlm-emo-t")
        auswertung = classifier2(user_text)
        self.result_output.delete(1.0, tk.END)
        self.result_output.insert(tk.END, str(auswertung))

    def clear_text(self):
        self.text_input.delete("1.0", tk.END)
        self.result_output.delete(1.0, tk.END)

    def open_youtube(self):
        webbrowser.open("https://www.youtube.com/results?search_query=")

#Wrapper fuer Button auf Seite 21
def huggingface_seite_oeffnen():
    #webbrowser.open("https://huggingface.co/models", new=1, autoraise=True)
    dcc.Link("Oeffne Hugging Face", href="https://huggingface.co/models", target="_blank")


'''
#Hier befindet sich der Client API, weil die Dash und Flask API scheinbar nicht in der Lage sind,
#!!! clientseitige Befehle auszufuehren.
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
'''

class ClientAPI:
    """ API fuer PyWebView, um externe Fenster clientseitig zu oeffnen """

    def open_tkinter(self):
        print("[Client] Oeffne Zeichenfenster...")
        threading.Thread(target=Zeichenfenster.open_drawing_window_in_process, daemon=True).start()

    def open_blanc_fenster(self):
        print("[Client] Oeffne BlancFenster...")
        threading.Thread(target=BlancFenster.open_drawing_window_in_process, daemon=True).start()

    def open_opencv_camera(self):
        print("[Client] Starte OpenCV-Kamera...")
        threading.Thread(target=CameraApp.open_camera_window_in_process, daemon=True).start()

    def start_audio_recording(self):
        print("[Client] Starte Audio-Aufnahme...")
        threading.Thread(target=AudioApp.start_recording, daemon=True).start()

    def stop_audio_recording(self):
        print("[Client] Stoppe Audio-Aufnahme...")
        threading.Thread(target=AudioApp.stop_recording, daemon=True).start()



'''
#Hier sind Buttons, Checkboxen, Textfelder und zugehoerige Mappings und so weiter definiert
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
'''

######Main-Abschnitt der Web-App

# Flask Server initialisieren
server = Flask(__name__)

# Verzeichnis mit den Bildern-
ASSETS_DIR = "./pages_5"

# Alle Dateien im Verzeichnis
files = os.listdir(ASSETS_DIR) if os.path.exists(ASSETS_DIR) else []
files = sorted([f for f in files if f.startswith("Aufgabe3") and f.endswith(".png")])
MAX_PAGES = len(files)  # Anzahl der Seiten basierend auf den Dateien

if not MAX_PAGES:
    raise FileNotFoundError("Keine gueltigen Seiten im Verzeichnis gefunden!")

# Kapitel-Mapping: Kapitelnummer -> Seitennummer
CHAPTER_MAPPING = {
    1: 4,
    2: 15,
    3: 23,
    4: 45,
    5: 72,
    6: 84,
    7: 99,
    8: 114,
    9: 121 
}

#Globale Variable, die die Dash Textfeld-Inputs zwischenspeichert
stored_text_data = None

def print_stored_text_data():
    print(get_text("text-94-1"))

# Globale Datenstruktur fuer Textfelder mit IDs
TEXT_FIELDS = {
    5: [
        {"id": "text-5-1", "x": 20, "y": 58.9, "width": 35, "height": 18.0},
        {"id": "text-5-2", "x": 35, "y": 82.2, "width": 35, "height": 18.0},
        {"id": "text-5-3", "x": 48, "y": 58.1, "width": 35, "height": 18.0},
        {"id": "text-5-4", "x": 48, "y": 82.1, "width": 35, "height": 18.0},
        {"id": "text-5-5", "x": 60, "y": 62.3, "width": 35, "height": 18.0},
        {"id": "text-5-6", "x": 62, "y": 82.4, "width": 35, "height": 18.0},
        {"id": "text-5-7", "x": 70, "y": 59.2, "width": 35, "height": 18.0},
    ],
    6: [
        {"id": "text-6-1", "x": 58, "y": 72.3, "width": 50, "height": 18.0},
        {"id": "text-6-2", "x": 48, "y": 77.5, "width": 50, "height": 18.0},
        {"id": "text-6-3", "x": 71, "y": 83, "width": 50, "height": 18.0},

        {"id": "text-6-4", "x": 68.2, "y": 42.8, "width": 32.5, "height": 15.0},
        {"id": "text-6-5", "x": 75.6, "y": 45.2, "width": 35, "height": 15.0},
        {"id": "text-6-6", "x": 44.6, "y": 50.5, "width": 36, "height": 15.0},
        {"id": "text-6-7", "x": 34.9, "y": 56, "width": 34, "height": 15.0},
        {"id": "text-6-8", "x": 52.3, "y": 56, "width": 36, "height": 15.0},
        {"id": "text-6-9", "x": 21.3, "y": 58.4, "width": 33.2, "height": 15.0},
    ],
    8: [
        {"id": "text-8-1", "x": 37, "y": 91, "width": 65, "height": 58.0},
    ],
    16: [
        {"id": "text-16-1", "x": 51, "y": 58, "width": 93, "height": 27.0},
        {"id": "text-16-2", "x": 51, "y": 71.5, "width": 93, "height": 27.0},
        {"id": "text-16-3", "x": 51, "y": 87, "width": 93, "height": 31.0},
        {"id": "text-16-4", "x": 51, "y": 103.4, "width": 93, "height": 31.0},
    ],
    20: [
        {"id": "text-17-1", "x": 51, "y": 61, "width": 93, "height": 27.0},
        {"id": "text-17-2", "x": 51, "y": 74.5, "width": 93, "height": 27.0},
        {"id": "text-17-3", "x": 51, "y": 90, "width": 93, "height": 31.0},
        {"id": "text-17-3", "x": 51, "y": 106.4, "width": 93, "height": 31.0},
    ],
    20: [
        {"id": "text-18-3", "x": 51.2, "y": 86, "width": 94.6, "height": 60.0},
    ],
    25: [
        {"id": "text-23-1", "x": 32, "y": 39.2, "width": 35, "height": 18.0},
        {"id": "text-23-2", "x": 32, "y": 42.2, "width": 35, "height": 18.0},
        {"id": "text-23-3", "x": 32, "y": 45.2, "width": 35, "height": 18.0},
        {"id": "text-23-4", "x": 32, "y": 47.8, "width": 35, "height": 18.0},
    ],
    27: [
        {"id": "text-25-1", "x": 83.5, "y": 55.4, "width": 25, "height": 18.0},
        {"id": "text-25-2", "x": 83.5, "y": 60.9, "width": 25, "height": 18.0},
        {"id": "text-25-3", "x": 83.5, "y": 66.4, "width": 25, "height": 18.0},
        {"id": "text-25-4", "x": 83.5, "y": 71.9, "width": 25, "height": 18.0},
        {"id": "text-25-5", "x": 83.5, "y": 76.4, "width": 25, "height": 18.0},
        {"id": "text-25-6", "x": 91, "y": 55.4, "width": 25, "height": 18.0},
        {"id": "text-25-7", "x": 91, "y": 60.9, "width": 25, "height": 18.0},
        {"id": "text-25-8", "x": 91, "y": 66.4, "width": 25, "height": 18.0},
        {"id": "text-25-9", "x": 91, "y": 71.9, "width": 25, "height": 18.0},
        {"id": "text-25-10", "x": 91, "y": 76.4, "width": 25, "height": 18.0},
    ],
    29: [
        {"id": "text-27-1", "x": 60.5, "y": 82.4, "width": 22, "height": 18.0},
        {"id": "text-27-2", "x": 60.5, "y": 86.9, "width": 22, "height": 18.0},
        {"id": "text-27-3", "x": 66.5, "y": 82.4, "width": 22, "height": 18.0},
        {"id": "text-27-4", "x": 66.5, "y": 86.9, "width": 22, "height": 18.0},
    ],
    30: [
        {"id": "text-28-1", "x": 70.5, "y": 33.8, "width": 22, "height": 18.0},
        {"id": "text-28-2", "x": 70.5, "y": 38.8, "width": 22, "height": 18.0},
        {"id": "text-28-3", "x": 70.5, "y": 43.8, "width": 22, "height": 18.0},
        {"id": "text-28-4", "x": 76.8, "y": 33.8, "width": 22, "height": 18.0},
        {"id": "text-18-5", "x": 76.8, "y": 38.8, "width": 22, "height": 18.0},
        {"id": "text-28-6", "x": 76.8, "y": 43.8, "width": 22, "height": 18.0},
        {"id": "text-28-7", "x": 83.1, "y": 33.8, "width": 22, "height": 18.0},
        {"id": "text-28-8", "x": 83.1, "y": 38.8, "width": 22, "height": 18.0},
        {"id": "text-28-9", "x": 83.1, "y": 43.8, "width": 22, "height": 18.0},
        {"id": "text-28-10", "x": 89.5, "y": 33.8, "width": 22, "height": 18.0},
        {"id": "text-28-11", "x": 89.5, "y": 38.8, "width": 22, "height": 18.0},
        {"id": "text-28-12", "x": 89.5, "y": 43.8, "width": 22, "height": 18.0},

        {"id": "text-28-13", "x": 69, "y": 58.3, "width": 22, "height": 18.0},
        {"id": "text-28-14", "x": 69, "y": 63.3, "width": 22, "height": 18.0},
        {"id": "text-28-15", "x": 69, "y": 68.3, "width": 22, "height": 18.0},
        {"id": "text-28-16", "x": 75.3, "y": 58.3, "width": 22, "height": 18.0},
        {"id": "text-18-17", "x": 75.3, "y": 63.3, "width": 22, "height": 18.0},
        {"id": "text-28-18", "x": 75.3, "y": 68.3, "width": 22, "height": 18.0},
        {"id": "text-28-19", "x": 81.6, "y": 58.3, "width": 22, "height": 18.0},
        {"id": "text-28-20", "x": 81.6, "y": 63.3, "width": 22, "height": 18.0},
        {"id": "text-28-21", "x": 81.6, "y": 68.3, "width": 22, "height": 18.0},
    ],
    35: [
        {"id": "text-33-1", "x": 48, "y": 40, "width": 30, "height": 18.0},
        {"id": "text-33-2", "x": 48, "y": 43.5, "width": 30, "height": 18.0},
        {"id": "text-33-3", "x": 48, "y": 47, "width": 30, "height": 18.0},
    ],
    37: [
        {"id": "text-37-1", "x": 30, "y": 43, "width": 30, "height": 18.0},
    ],
    42: [
        {"id": "text-42-1", "x": 33, "y": 56.2, "width": 30, "height": 18.0},
    ],
    43: [
        {"id": "text-43-1", "x": 52.5, "y": 37.5, "width": 95, "height": 30.0},

        {"id": "text-43-2", "x": 33, "y": 82, "width": 25, "height": 18.0},
        {"id": "text-43-3", "x": 41, "y": 82, "width": 25, "height": 18.0},
        {"id": "text-43-4", "x": 49, "y": 82, "width": 25, "height": 18.0},
        {"id": "text-43-5", "x": 57, "y": 82, "width": 25, "height": 18.0},
        {"id": "text-43-6", "x": 65, "y": 82, "width": 25, "height": 18.0},
    ],
    49: [
        {"id": "text-49-1", "x": 52.5, "y": 43, "width": 95, "height": 35.0},
        {"id": "text-49-2", "x": 52.5, "y": 67.5, "width": 95, "height": 45.0},
    ],
    50: [
        {"id": "text-50-3", "x": 52.5, "y": 40, "width": 95, "height": 35.0},
        {"id": "text-50-4", "x": 52.5, "y": 62, "width": 95, "height": 45.0},
    ],
    51: [
        {"id": "text-51-1", "x": 43.5, "y": 59, "width": 80, "height": 60.0},
    ],
    65: [
        {"id": "text-65-1", "x": 52, "y": 80.5, "width": 95, "height": 55.0},
    ],
    69: [
        {"id": "text-69-1", "x": 65, "y": 66, "width": 77.5, "height": 16.0},
        {"id": "text-69-2", "x": 65, "y": 71.5, "width": 77.5, "height": 16.0},
        {"id": "text-69-3", "x": 67, "y": 78, "width": 76, "height": 18.0},
        {"id": "text-69-4", "x": 51.4, "y": 103, "width": 95, "height": 25.0},
    ],
    93: [
        {"id": "text-93-1", "x": 55, "y": 39.5, "width": 23, "height": 18.0},
        {"id": "text-93-2", "x": 23.3, "y": 47.7, "width": 23, "height": 18.0},
        {"id": "text-93-3", "x": 52, "y": 56, "width": 23, "height": 18.0},
        {"id": "text-93-4", "x": 49, "y": 64, "width": 23, "height": 18.0},
        {"id": "text-93-5", "x": 47, "y": 72, "width": 23, "height": 18.0},
        {"id": "text-93-6", "x": 64, "y": 80, "width": 23, "height": 18.0},
        {"id": "text-93-7", "x": 46, "y": 98, "width": 80, "height": 20.0},
    ],
    94: [
        {"id": "text-94-1", "x": 40, "y": 38, "width": 70, "height": 30.0},
        {"id": "text-94-2", "x": 88, "y": 80, "width": 25, "height": 17.0},

        {"id": "text-94-3", "x": 40, "y": 50, "width": 70, "height": 26.0},
    ],
    96: [
        {"id": "text-96-1", "x": 29.5, "y": 72, "width": 20, "height": 18.0},
        {"id": "text-96-2", "x": 37.5, "y": 72, "width": 20, "height": 18.0},
        {"id": "text-96-3", "x": 45.5, "y": 72, "width": 20, "height": 18.0},
        {"id": "text-96-4", "x": 54, "y": 72, "width": 20, "height": 18.0},
    ],
}

CHECK_BOXES = {
    13: [
        {"id": "checkbox-13-1", "x": 10, "y": 80.5, "scale": 2.5, "checked": False},
        {"id": "checkbox-13-2", "x": 10, "y": 83.2, "scale": 2.5, "checked": False},
        {"id": "checkbox-13-3", "x": 10, "y": 86, "scale": 2.5, "checked": False},
        {"id": "checkbox-13-4", "x": 10, "y": 88.8, "scale": 2.5, "checked": False},
    ],
    42: [
        {"id": "checkbox-42-1", "x": 50, "y": 23.5, "scale": 2.5, "checked": False},
        {"id": "checkbox-42-2", "x": 50, "y": 26.2, "scale": 2.5, "checked": False},
        {"id": "checkbox-42-3", "x": 50, "y": 29, "scale": 2.5, "checked": False},
        {"id": "checkbox-42-4", "x": 50, "y": 31.8, "scale": 2.5, "checked": False},
    ],
    43: [
        {"id": "checkbox-43-1", "x": 60, "y": 37, "scale": 2.5, "checked": False},
        {"id": "checkbox-43-2", "x": 60, "y": 39.8, "scale": 2.5, "checked": False},
        {"id": "checkbox-43-3", "x": 60, "y": 42.7, "scale": 2.5, "checked": False},
    ],
    83: [
        {"id": "checkbox-83-1", "x": 34, "y": 18, "scale": 2.5, "checked": False},
        {"id": "checkbox-83-2", "x": 15, "y": 23.5, "scale": 2.5, "checked": False},
        {"id": "checkbox-83-3", "x": 50, "y": 29, "scale": 2.5, "checked": False},

        {"id": "checkbox-83-4", "x": 50, "y": 37, "scale": 2.5, "checked": False},
        {"id": "checkbox-83-5", "x": 50, "y": 40, "scale": 2.5, "checked": False},
        {"id": "checkbox-83-6", "x": 50, "y": 43, "scale": 2.5, "checked": False},
    ],
    94: [
        {"id": "checkbox-94-1", "x": 86, "y": 53.5, "scale": 2.5, "checked": False},
        {"id": "checkbox-94-2", "x": 50, "y": 59, "scale": 2.5, "checked": False},
        {"id": "checkbox-94-3", "x": 86, "y": 61.7, "scale": 2.5, "checked": False},
        {"id": "checkbox-94-4", "x": 42, "y": 64.2, "scale": 2.5, "checked": False},

        {"id": "checkbox-94-5", "x": 50, "y": 83.2, "scale": 2.5, "checked": False},
        {"id": "checkbox-94-6", "x": 50, "y": 86, "scale": 2.5, "checked": False},
        {"id": "checkbox-94-7", "x": 50, "y": 88.7, "scale": 2.5, "checked": False},
    ],
    96: [
        {"id": "checkbox-93-1", "x": 70, "y": 23.3, "scale": 2.5, "checked": False},
        {"id": "checkbox-93-2", "x": 70, "y": 26.5, "scale": 2.5, "checked": False},
        {"id": "checkbox-93-3", "x": 70, "y": 28.7, "scale": 2.5, "checked": False},
        {"id": "checkbox-93-4", "x": 70, "y": 31.7, "scale": 2.5, "checked": False},

        {"id": "checkbox-93-5", "x": 75, "y": 45.3, "scale": 2.5, "checked": False},
        {"id": "checkbox-93-6", "x": 87, "y": 48, "scale": 2.5, "checked": False},
        {"id": "checkbox-93-7", "x": 13, "y": 53.3, "scale": 2.5, "checked": False},
        {"id": "checkbox-93-8", "x": 33, "y": 58.8, "scale": 2.5, "checked": False},
    ],
}

#wrapper fuer die beiden methoden bei Visualisierung und Update des Graphen
visualization_active = False
visualization_thread = None  # Neuer Thread fuer Visualisierung
def update_visualization():
    """Externe Methode zur Aktualisierung der Visualisierung."""
    global visualization_active
    while visualization_active:
        q_data = AudioApp.get_queue_data_10()
        if q_data:
            print(f"[DEBUG] Visualisierung aktualisiert mit: {q_data}")
        time.sleep(0.05)  # Aktualisierung alle 50ms


def control_visualization(start_clicks, stop_clicks, current_state):
    """Steuert die Visualisierung ueber externe Buttons."""
    global visualization_active
    if ctx.triggered_id == 'button-17-1':  # Start-Button
        if not visualization_active:
            visualization_active = True
            threading.Thread(target=update_visualization, daemon=True).start()
            return True
    elif ctx.triggered_id == 'button-17-2':  # Stop-Button
        visualization_active = False
        return False
    return dash.no_update

def start_recording_and_update():
    AudioApp.start_recording()
    update_visualization()


AUDIO_VISUALISATIONS = {
    18: {  # Live Audio Visualisierung nur fuer Seite 17
        "id": "audio-17-1",
        "x": 49,  # Position X in Prozent
        "y": 16,  # Position Y in Prozent
        "width": 90,  # Breite in Prozent
        "height": 10,  # Hoehe in Prozent
    }
}




BUTTONS = {
    16: [
        {"id": "button-16-1", "x": 70, "y": 12, "label": "Klick mich", "color": "primary", "background_color" : "grey" , "method": CameraApp._run_tkinter},
    ],

    18: [
        {"id": "button-18-1", "x": 50, "y": 22.5, "label": "Audio Aufnehmen", "color": "success", "background_color" : "grey" , "method": start_recording_and_update},
        {"id": "button-18-2", "x": 50, "y": 28.5, "label": "Audio Beenden", "color": "success", "background_color" : "grey" , "method": AudioApp.stop_recording},
    ],
    20: [
        {"id": "button-20-1", "x": 5, "y": 15, "label": "OK", "color": "success", "background_color" : "grey" , "method": YoutubeKommentarApp._run_tkinter},
    ],
    21: [
        {"id": "button-21-1", "x": 20.1, "y": 13.1, "label": "\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003\u2003", "color": "blue", "background_color" : "rgba(128,128,128,0.5)" , "method": huggingface_seite_oeffnen},
    ],
    33: [
        {"id": "button-33-1", "x": 12.5, "y": 18.5, "label": "Zeichenfenster oeffnen", "color": "blue", "background_color" : "grey" , "method": Zeichenfenster.open_drawing_window_in_process},
    ],
    42: [
        {"id": "button-42-1", "x": 12.5, "y": 43.5, "label": "Zeichenfenster oeffnen", "color": "blue", "background_color" : "grey" , "method": Zeichenfenster.open_drawing_window_in_process},
    ],
    43: [
        {"id": "button-43-1", "x": 10, "y": 83.5, "label": "Fenster oeffnen", "color": "blue", "background_color" : "grey" , "method": BlancFenster.open_drawing_window_in_process},
    ],
    69: [
        {"id": "button-69-1", "x": 12.5, "y": 83.5, "label": "Zeichenfenster oeffnen", "color": "blue", "background_color" : "grey" , "method": Zeichenfenster.open_drawing_window_in_process},
    ],
    94: [
        {"id": "button-94-1", "x": 10, "y": 34.5, "label": "Analysieren", "color": "blue", "background_color" : "grey" , "method": print_stored_text_data},
    ],
}

'''
#Hier ist das Dash-Layout definiert, inklusive Callbacks an die Dash-App, und was sonst noch so erzwungenermassen
#!!!dazugehoert.
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
'''

# Initialisiere die Dash-App
app = Dash(__name__, server=server, assets_folder=ASSETS_DIR, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])##LETZER PARAMETER WURDE UEBERARBEITET

# Layout der App
app.layout = html.Div([
    dcc.Store(id='text-store', data={}, storage_type='memory'),  # Speichert alle Texteingaben
    dcc.Location(id='url', refresh=False),
    #dcc.Interval(id='interval-update', interval=50, n_intervals=0),  # Alle 50 ms

    # Dummy-Output fuer den Button-Callback
    html.Div(id='dummy-output', style={'display': 'none'}),
    html.Button(id='dummy-trigger', style={'display': 'none'}, n_clicks=0),

    html.Div([
        # Linker Bereich (Seite und Kapitel auswaehlen)
        html.Div([
            html.Div([
                dcc.Input(id='page-input', type='number', min=1, max=MAX_PAGES, placeholder='Seite', style={
                    'width': '70px',
                    'marginRight': '10px',
                    'padding': '5px',
                    'fontSize': '16px',
                    'border': '1px solid #ccc',
                    'borderRadius': '3px'
                }),
                html.Button('Springen', id='jump-button', style={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontSize': '16px',
                    'borderRadius': '3px',
                    'cursor': 'pointer'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '20px'}),
            html.Div([
                dcc.Input(id='chapter-input', type='number', min=1, max=len(CHAPTER_MAPPING), placeholder='Kapitel', style={
                    'width': '70px',
                    'marginRight': '10px',
                    'padding': '5px',
                    'fontSize': '16px',
                    'border': '1px solid #ccc',
                    'borderRadius': '3px'
                }),
                html.Button('Zu Kapitel', id='chapter-button', style={
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 20px',
                    'fontSize': '16px',
                    'borderRadius': '3px',
                    'cursor': 'pointer'
                })
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'flex': '1', 'textAlign': 'left', 'paddingLeft': '20px', 'display': 'flex'}),
        html.Div([
            html.A('Zurueck', id='back-button', href='/1', style={
                'marginRight': '15px',
                'backgroundColor': '#ddd',
                'color': 'black',
                'border': '1px solid #ccc',
                'padding': '10px 20px',
                'textAlign': 'center',
                'textDecoration': 'none',
                'display': 'inline-block',
                'fontSize': '16px',
                'borderRadius': '3px',
                'cursor': 'pointer'
            }),
            html.A('Weiter', id='forward-button', href='/2', style={
                'backgroundColor': '#ddd',
                'color': 'black',
                'border': '1px solid #ccc',
                'padding': '10px 20px',
                'textAlign': 'center',
                'textDecoration': 'none',
                'display': 'inline-block',
                'fontSize': '16px',
                'borderRadius': '3px',
                'cursor': 'pointer'
            })
        ], style={'flex': '1', 'textAlign': 'center'}),
        html.Div([], style={'flex': '1', 'textAlign': 'right', 'paddingRight': '20px'})
    ], style={
        'position': 'fixed',
        'top': '0',
        'width': '100%',
        'background': 'white',
        'display': 'flex',
        'alignItems': 'center',
        'padding': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'zIndex': '1000',
    }),
    html.Div(id='page-content', style={'marginTop': '50px', 'width': '100%', 'height': 'calc(100vh - 50px)'})
])

'''# WebSocket Client direkt in der Main-Datei
sio = socketio.Client()

# Verbindung zum WebSocket-Server (ersetze mit deiner Dash-Server-IP)
SERVER_URL = "http://0.0.0.0:8050"  # Falls Dash auf einem anderen Rechner laeuft, IP anpassen
sio.connect(SERVER_URL)'''

'''
Hier in diesem Abschnitt sind alle Callbacks definiert, die Daten an die eigentliche Dash-App senden und Daten empfangen.
'''
@app.callback(
    Output('text-store', 'data'),
    Input({'type': 'text-field', 'index': dash.ALL}, 'value'),  # Alle Eingaben gleichzeitig ueberwachen
    State('text-store', 'data'),
    prevent_initial_call=True
)
def save_texts(values, stored_data):
    """Speichert Werte nur, wenn sich etwas geaendert hat."""
    stored_data = stored_data or {}

    has_changed = False
    for idx, val in enumerate(values):
        key = f"text-{idx}"
        if stored_data.get(key) != val:
            stored_data[key] = val
            has_changed = True
    
    if not has_changed:
        raise PreventUpdate  # Verhindert unnoetige Updates

    global stored_text_data 
    stored_text_data = stored_data

    print("Gespeicherte Daten:", stored_data)  # Debugging
    return stored_data

@app.callback(
    [Output(field['id'], 'value') for page in TEXT_FIELDS for field in TEXT_FIELDS[page]],  # Setzt alle Textfelder
    Input('text-store', 'data')  # Holt gespeicherte Werte
)
def load_texts(stored_data):
    if not stored_data:
        raise PreventUpdate
    
    # Rueckgabe der gespeicherten Werte fuer jedes Feld
    return [stored_data.get(field['id'], '') for page in TEXT_FIELDS for field in TEXT_FIELDS[page]]


@app.callback(
    Output("text_output", "children"),
    Input("text_id_input", "value"),
    State("text-store", "data")
)
def get_text(text_id):
    """
    Holt den gespeicherten Text fuer eine bestimmte ID direkt aus dem Dash Store.
    
    :param text_id: Die ID des gewuenschten Textfeldes
    :return: Der gespeicherte Text oder ein leerer String, falls nicht vorhanden.
    """
    stored_data = dcc.Store(id="text-store").data if hasattr(dcc.Store(id="text-store"), "data") else {}
    print("Folgender Text wird gespeichert: ", stored_data)
    if not text_id or not stored_data:
        return "Kein Text vorhanden"
    
    return stored_data.get(text_id, "")

# Callback, um den Graphen bei neuen Audio-Daten zu aktualisieren
@app.callback(
    Output('audio-waveform-graph', 'figure'),
    Input('interval-update', 'n_intervals')  # Der Timer, der alle 50 ms getriggert wird
)
def update_audio_graph(n_intervals):
    # Ueberpruefen, ob die Aufnahme laeuft (AudioApp.audio_flag ist True)
    if not AudioApp.audio_flag:
        return dash.no_update  # Kein Update des Graphen, wenn die Aufnahme nicht laeuft

    # Falls die Flagge True ist, die neuen y-Werte aus der AudioApp holen
    y_values = AudioApp.audio_list  # Hier holen wir uns die aktuellen y-Werte aus AudioApp.audio_list

    # Aktuellen Graphen zurueckgeben
    figure = {
        "data": [
            go.Scatter(
                x=list(range(50)),  # x-Werte bleiben gleich
                y=y_values,          # Dynamische y-Werte
                mode="lines",
                line=dict(color="blue", width=2)  # Farbe + Linienstaerke
            )
        ],
        "layout": go.Layout(
            xaxis=dict(visible=False),  # X-Achse ausblenden
            yaxis=dict(visible=False),  # Y-Achse ausblenden
            plot_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0)  # Rand entfernen
        )
    }
    return figure


#Callback fuer das springen zwischen den Seiten via Jump-Button oder der Navigationsleiste
@app.callback(
    Output('url', 'pathname'),
    [Input('jump-button', 'n_clicks'),
     Input('chapter-button', 'n_clicks')],
    [State('page-input', 'value'),
     State('chapter-input', 'value')]
)
def jump_to_page_or_chapter(jump_clicks, chapter_clicks, page_num, chapter_num):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'jump-button' and page_num:
        return f'/{max(1, min(MAX_PAGES, page_num))}'
    elif button_id == 'chapter-button' and chapter_num in CHAPTER_MAPPING:
        return f'/{CHAPTER_MAPPING[chapter_num]}'
    return no_update

@app.callback(
    [Output('page-content', 'children'),
     Output('back-button', 'href'),
     Output('forward-button', 'href')],
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if not pathname or pathname == "/":
        page_num = 1
    else:
        try:
            page_num = int(pathname.split('/')[-1])
        except ValueError:
            page_num = 1
    page_num = max(1, min(MAX_PAGES, page_num))

    back_href = f'/{page_num - 1}' if page_num > 1 else '/1'
    forward_href = f'/{page_num + 1}' if page_num < MAX_PAGES else f'/{MAX_PAGES}'

    return [create_page(page_num), back_href, forward_href]


#Callback fuer die anderen Buttons, die innerhalb der Aufgaben usw eingefuegt wurden
@app.callback(
    Output('dummy-output', 'children'),
    Input({'type': 'dynamic-button', 'index': dash.ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def button_callback(n_clicks):
    ctx = callback_context

    if not ctx.triggered:
        raise PreventUpdate  # Kein Button wurde geklickt

    # Finde heraus, welcher Button geklickt wurde
    triggered_button = ctx.triggered[0]['prop_id'].split('.')[0]
    button_id = eval(triggered_button)  # {'type': 'dynamic-button', 'index': XYZ}

    # Falls n_clicks None oder 0 ist, soll nichts passieren
    if not any(n_clicks):  
        raise PreventUpdate  

    print(f"Button {button_id['index']} wurde geklickt!")  # Debug-Output

    # Suche den geklickten Button in der BUTTONS-Datenstruktur
    for page, buttons in BUTTONS.items():
        for button in buttons:
            if button["id"] == button_id['index']:
                method = button["method"]
                if callable(method):
                    print(f"Starte Methode {method.__name__}...")
                    threading.Thread(target=method).start()  # Starte Methode in eigenem Thread
                return no_update

    return no_update




# Globale Variable zur Steuerung der Visualisierung, vermutlich redundant und kann rausfliegen
visualization_active = False

# DCC Store zum Speichern des Status; vermutlich redundanter Code, der nachtraeglich rausfliegt
app.layout.children.append(dcc.Store(id='visualization-store', data=False))


'''
#Die create-pages methode generiert die ganzen Buttons und so weiter dynamisch anhand der Parameter,
#!!! die ich oben in den Textfeld-Button-Listen festgelegt habe.
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
'''
'''Hier ist alles festgelegt, was das Erstellen der Einzelseiten betrifft. 
Die Buttons/Textfelder/Ankreuzkaestchen werden in dieser methode eingefuegt.
Weiterhin wird das Updaten des Audio-Graphen hierdrin durchgefuehrt bzw. der Div dafuer geschaffen.
'''
def create_page(page_num):
    text_fields = []
    if page_num in TEXT_FIELDS:
        for field in TEXT_FIELDS[page_num]:
            text_fields.append(
                html.Div(
                    dcc.Textarea(
                        id=field["id"],
                        placeholder="",
                        style={
                            'width': f"{field['width']}%",  
                            'height': f"{field['height']}%",  
                            'padding': '10px',
                            'fontSize': '16px',
                            'border': '1px solid #ccc',
                            'borderRadius': '5px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                            'resize': 'none',  
                            'pointerEvents': 'auto',  
                        }
                    ),
                    style={
                        "position": "absolute",
                        "top": f"{field['y']}%",  
                        "left": f"{field['x']}%",  
                        "width": f"{field['width']}%",  
                        "height": f"{field['height']}%",  
                        "transform": "translate(-50%, -50%)",  
                    }
                )
            )

    # Automatisch alle Checkboxen generieren, die in CHECK_BOXES definiert sind
    check_boxes = []
    if page_num in CHECK_BOXES:
        for box in CHECK_BOXES[page_num]:
            check_boxes.append(
                html.Div(
                    dcc.Checklist(
                        id=box["id"],
                        options=[{'label': '', 'value': 'checked'}],
                        value=['checked'] if box["checked"] else [],
                        style={
                            'pointerEvents': 'auto'  # Interaktion erlauben
                        },
                        inputStyle={
                            'transform': f'scale({box["scale"]})',  # Skalierung
                            'transformOrigin': 'top left',
                            'margin': '10px'
                        }
                    ),
                    style={
                        "position": "absolute",
                        "top": f"{box['y']}%",  # Abstand von oben in Prozent
                        "left": f"{box['x']}%",  # Abstand von links in Prozent
                        "transform": "translate(-50%, -50%)"  # Zentriert die Checkbox
                    }
                )
            )

    buttons = []
    if page_num in BUTTONS:
        for button in BUTTONS[page_num]:
            buttons.append(
                html.Div(
                    html.Button(
                        button["label"], 
                        id={'type': 'dynamic-button', 'index': button["id"]},  # Hier dynamische ID!
                        n_clicks=0,
                        style={
                            'backgroundColor': button.get("background_color"),#'grey',
                            'color': 'black',
                            'border': 'none',
                            'padding': '10px 20px',
                            'fontSize': '16px',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }
                    ),
                    style={
                        "position": "absolute",
                        "top": f"{button['y']}%",  
                        "left": f"{button['x']}%",  
                        "transform": "translate(-50%, -50%)"  
                    }
                )
            )
    
    
    # Dummy-Daten fuer die Linie des Audiovisualisierungsgraphen auf Seite 17
    x_values = list(range(50))  # X-Achse von 0 bis 9
    y_values = [0]*50
    # Audio-Visualisierung mit Wellenform
    audio_div = None
    if page_num in AUDIO_VISUALISATIONS:
        audio_data = AUDIO_VISUALISATIONS[page_num]

        audio_div = html.Div([
            html.Div([
                #Intervall fuer das aktualisieren des graphen
                dcc.Interval(id='interval-update', interval=50, n_intervals=0),  # Alle 50 ms
                # Graph ohne Achsen
                dcc.Graph(
                    id="audio-waveform-graph",
                    figure={
                        "data": [
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode="lines",
                                line=dict(color="blue", width=2)  # Farbe + Linienstaerke
                            )
                        ],
                        "layout": go.Layout(
                            xaxis=dict(visible=False),  # X-Achse ausblenden
                            yaxis=dict(visible=False, range=[-0.01, 0.01]),  # Y-Achse ausblenden
                            plot_bgcolor="white",
                            margin=dict(l=0, r=0, t=0, b=0)  # Rand entfernen
                        )
                    },
                    style={"width": "100%", "height": "100%"}
                )
            ], style={
                "width": "100%",
                "height": "100%",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center"
            })
        ], style={
            "position": "absolute",
            "top": f"{audio_data['y']}%",
            "left": f"{audio_data['x']}%",
            "width": f"{audio_data['width']}%",
            "height": f"{audio_data['height']}%",
            "border": "2px solid black",
            "backgroundColor": "#f0f0f0",
            "textAlign": "center",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "pointerEvents": "auto",
            "transform": "translate(-50%, -50%)"
        })
    
    #hier in dem return statement wird das Div seitenuebergreifend einheitlich gestaltet und zurueckgegeben
    return html.Div([
        html.Div(
            style={'position': 'relative', 'width': '80%', 'margin': '0 auto'},
            children=[
                html.Div(
                    style={
                        "position": "relative",
                        "width": "100%",  
                        "height": "auto",
                        "overflow": "hidden",
                    },
                    children=[
                        html.Img(
                            src=f"/assets/{files[page_num - 1]}",
                            style={
                                'width': '100%',
                                'height': 'auto',
                                'display': 'block',
                                'margin': '0 auto'
                            }
                        ),
                        html.Div(
                            style={
                                "position": "absolute",
                                "top": "0px",
                                "left": "0px",
                                "width": "100%",
                                "height": "100%",
                                "pointerEvents": "auto",
                            },
                            children=text_fields + check_boxes + buttons + ([audio_div] if audio_div else [])
                        )
                    ]
                )
            ]
        )
    ])

'''
#Paar Debug-Hilfsmethoden
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
'''
#redundante Methode... nur zu Testzwecken, ob die Audiodaten aus der AudioApp gezogen werden koennen
def start_queue_logging():
    """ Laeuft in einem separaten Thread und ruft jede Sekunde print_queue_data auf """
    while True:
        #get_current_page(ctx)
        #AudioApp.print_queue_data()  # Jetzt OHNE Endlosschleife
        time.sleep(1)

def debugging_print():
    while True:
        #print("Hallo")
        print("hallo", flush=True)
        time.sleep(1)



'''
#Hier Start von Dash und von der Client-API
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
#!!!
'''
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)



#hier auch... zu Testzwecken, redundant
'''if __name__ == '__main__':
    #Zeichenfenster._run_tkinter()
    app.run_server(debug=True)'''

