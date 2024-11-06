import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import json
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List
import os
import numpy as np

from backend import detection_backend

class VideoCapture:
    """Bufferless VideoCapture class to prevent frame lag"""
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue(maxsize=1)
        self._stop = False
        self._thread = threading.Thread(target=self._reader)
        self._thread.daemon = True
        self._thread.start()

    def _reader(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.full():
                self.q.put(frame)
            else:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put(frame)

    def read(self) -> Optional[np.ndarray]:
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self._stop = True
        self._thread.join()
        self.cap.release()

class POSApp:
    def __init__(self, root):
        self.root = root
        self.setup_config()
        self.setup_ui()
        self.setup_queues()
        self.setup_cameras()
        self.start_processing()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_config(self):
        self.colors = {
            'primary': '#be1d37',
            'secondary': '#f2f2f2',
            'text': '#ffffff'
        }
        self.camera_size = (640, 480)
        self.sidebar_width = 240
        self.update_interval = 33  # ~30 FPS
        self.detection_timeout = 2.0
        
        self.root.title('Tasty Fresh POS System')
        self.root.configure(bg=self.colors['primary'])
        self.root.minsize(950, 720)

    def setup_queues(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=1)
        self.cart_lock = threading.Lock()
        self.cart_items = []
        self.last_detection_time = time.time()
        self.detection_running = True

    def setup_ui(self):
        # Main layout
        self.create_icon()
        self.create_title()
        self.create_main_frame()
        self.create_sidebar()
        self.create_cart()
        self.create_status_bar()

    def create_icon(self):
        try:
            icon = tk.PhotoImage(file='./Tasty_Fresh_Icon.png')
            self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Could not load icon: {e}")
    
    def create_title(self):
        self.title_frame = tk.Frame(self.root, bg=self.colors['secondary'])
        self.title_frame.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        try:
            title_image = Image.open('./Tasty_Fresh_Title.png')
            self.title_photo = ImageTk.PhotoImage(title_image)
            self.title_label = tk.Label(
                self.title_frame, 
                image=self.title_photo, 
                bg=self.colors['secondary'],
                bd=0,
                highlightthickness=0
            )
            self.title_label.pack()
        except Exception as e:
            print(f"Could not load title image: {e}")
            # Fallback to text title
            tk.Label(
                self.title_frame,
                text="Tasty Fresh POS",
                font=('Helvetica', 24, 'bold'),
                bg=self.colors['secondary'],
                fg=self.colors['primary']
            ).pack()

    def create_main_frame(self):
        self.main_frame = tk.Frame(
            self.root,
            width=self.camera_size[0],
            height=self.camera_size[1],
            bg='black',
            bd=2,
            relief='solid'
        )
        self.main_frame.grid(row=1, column=0, padx=10, pady=10)
        self.main_camera_label = tk.Label(self.main_frame)
        self.main_camera_label.grid(row=0, column=0)

    def create_sidebar(self):
        self.sidebar = tk.Frame(
            self.root,
            width=self.sidebar_width,
            bg=self.colors['primary']
        )
        self.sidebar.grid(row=1, column=1, sticky='n', padx=10, pady=10)

        # Secondary camera
        cam_width = self.sidebar_width
        cam_height = int(cam_width * 3/4)
        self.secondary_frame = tk.Frame(
            self.sidebar,
            width=cam_width,
            height=cam_height,
            bg='gray',
            bd=2,
            relief='solid'
        )
        self.secondary_frame.grid(row=0, column=0, padx=10, pady=5)
        self.secondary_frame.grid_propagate(False)
        
        self.secondary_camera_label = tk.Label(self.secondary_frame)
        self.secondary_camera_label.grid(row=0, column=0)

    def create_cart(self):
        # Cart frame with scrollbar
        cart_frame = tk.Frame(self.sidebar, bg=self.colors['secondary'])
        cart_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        self.cart_canvas = tk.Canvas(
            cart_frame,
            width=self.sidebar_width-25,
            height=250,
            bg=self.colors['secondary'],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(cart_frame, orient='vertical', command=self.cart_canvas.yview)
        
        self.cart_items_frame = tk.Frame(self.cart_canvas, bg=self.colors['secondary'])
        self.cart_canvas.create_window((0, 0), window=self.cart_items_frame, anchor='nw', width=self.sidebar_width-45)
        
        self.cart_canvas.configure(yscrollcommand=scrollbar.set)
        self.cart_canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        self.cart_items_frame.bind('<Configure>', lambda e: self.cart_canvas.configure(scrollregion=self.cart_canvas.bbox('all')))
        
        # Total label
        self.total_label = tk.Label(
            self.sidebar,
            text='Total: $0.00',
            font=('Helvetica', 14, 'bold'),
            bg=self.colors['primary'],
            fg=self.colors['text']
        )
        self.total_label.grid(row=2, column=0, pady=10)

    def create_status_bar(self):
        self.status_bar = tk.Label(
            self.root,
            text="System Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg=self.colors['secondary'],
            fg=self.colors['primary']
        )
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky='ew')

    def setup_cameras(self):
        try:
            self.cap_main = VideoCapture(0)  # Main camera
            self.cap_secondary = VideoCapture(1)  # Secondary camera
        except Exception as e:
            self.status_bar.config(text=f"Camera Error: {str(e)}")

    def start_processing(self):
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        self.update_main_camera()
        self.update_secondary_camera()
        self.update_cart()

    def _detection_worker(self):
        while self.detection_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Get detections from backend
                detections = detection_backend.detect(img)
                
                with self.cart_lock:
                    if detections:
                        self.cart_items = detections
                        self.last_detection_time = time.time()
                    
                    if not self.detection_queue.full():
                        self.detection_queue.put((frame, detections))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.status_bar.config(text=f"Detection error: {str(e)[:50]}...")

    def update_main_camera(self):
        try:
            frame = self.cap_main.read()
            if frame is not None:
                frame = cv2.resize(frame, self.camera_size)
                
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame.copy())
                
                # Draw detections
                display_frame = frame.copy()
                with self.cart_lock:
                    if time.time() - self.last_detection_time <= self.detection_timeout:
                        for item in self.cart_items:
                            x1, y1, x2, y2 = map(int, item['bbox'])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, item['item'], (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.main_camera_label.imgtk = imgtk
                self.main_camera_label.configure(image=imgtk)
        
        except Exception as e:
            self.status_bar.config(text=f"Camera error: {str(e)}")
        
        self.root.after(self.update_interval, self.update_main_camera)

    def update_secondary_camera(self):
        try:
            frame = self.cap_secondary.read()
            if frame is not None:
                cam_width = self.sidebar_width - 20
                cam_height = int(cam_width * 3/4)
                frame = cv2.resize(frame, (cam_width, cam_height))
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.secondary_camera_label.imgtk = imgtk
                self.secondary_camera_label.configure(image=imgtk)
        
        except Exception as e:
            pass
        
        self.root.after(self.update_interval, self.update_secondary_camera)

    def update_cart(self):
        with self.cart_lock:
            # Clear existing items
            for widget in self.cart_items_frame.winfo_children():
                widget.destroy()

            # Headers
            self._create_cart_header()

            # Items
            total = 0
            for item in self.cart_items:
                price = float(item.get('price', 0))
                total += price
                self._create_cart_item(item['item'], price)

            # Update total
            self.total_label.config(text=f'Total: ${total:.2f}')

        self.root.after(500, self.update_cart)

    def _create_cart_header(self):
        header_frame = tk.Frame(self.cart_items_frame, bg=self.colors['secondary'])
        header_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=(5, 10))
        header_frame.grid_columnconfigure(0, weight=3)
        header_frame.grid_columnconfigure(1, weight=1)
        
        tk.Label(
            header_frame,
            text='Item',
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['secondary'],
            fg=self.colors['primary'],
            anchor='w'
        ).grid(row=0, column=0, sticky='w', padx=5)

        tk.Label(
            header_frame,
            text='Price',
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['secondary'],
            fg=self.colors['primary'],
            anchor='e'
        ).grid(row=0, column=1, sticky='e', padx=5)

    def _create_cart_item(self, name, price):
        row = len(self.cart_items_frame.winfo_children())
        
        # Item container with word wrap
        item_frame = tk.Frame(self.cart_items_frame, bg=self.colors['secondary'])
        item_frame.grid(row=row, column=0, sticky='ew', padx=5, pady=2)
        item_frame.grid_columnconfigure(0, weight=3)
        item_frame.grid_columnconfigure(1, weight=1)
        
        # Create wrapped label for item name
        name_label = tk.Label(
            item_frame,
            text=name,
            font=('Helvetica', 11),
            bg=self.colors['secondary'],
            fg=self.colors['primary'],
            anchor='w',
            justify=tk.LEFT,
            wraplength=self.sidebar_width - 100  # Leave space for price
        )
        name_label.grid(row=0, column=0, sticky='w', padx=5)

        # Price label
        price_label = tk.Label(
            item_frame,
            text=f'${price:.2f}',
            font=('Helvetica', 11),
            bg=self.colors['secondary'],
            fg=self.colors['primary'],
            anchor='e'
        )
        price_label.grid(row=0, column=1, sticky='e', padx=5)

    def on_closing(self):
        self.detection_running = False
        if hasattr(self, 'cap_main'):
            self.cap_main.stop()
        if hasattr(self, 'cap_secondary'):
            self.cap_secondary.stop()
        self.root.destroy()

def main():
    root = tk.Tk()
    
    # Center window
    window_size = (900, 700)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_size[0]) // 2
    y = (screen_height - window_size[1]) // 2
    root.geometry(f'{window_size[0]}x{window_size[1]}+{x}+{y}')
    
    app = POSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()