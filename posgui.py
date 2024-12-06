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

from backend import detection_backend  # Import our optimized backend

class VideoCapture:
    """Bufferless VideoCapture class to prevent frame lag"""
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue(maxsize=3)  # Limit queue size
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
            # Clear queue if full
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
        
        # Initialize queues and locks
        self.detection_queue = queue.Queue(maxsize=1)
        self.frame_queue = queue.Queue(maxsize=1)
        self.cart_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
        # Detection thread control
        self.detection_running = True
        
        # Store the last detection results
        self.last_detections = []
        
        # Maintain persistent cart state
        self.cart_state = []
        self.last_detection_time = time.time()
        self.detection_timeout = 0.5  # Time in seconds before removing items
        
        # Track displayed items to prevent unnecessary updates
        self.displayed_items = []
        
        # Store the last annotated frame
        self.last_annotated_frame = None
        
        # App configuration
        self.setup_app_config()
        
        # Initialize UI components
        self.setup_ui()
        
        # Initialize video capture
        self.setup_cameras()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        
        # Start camera updates
        self.schedule_camera_updates()
        
        # Start cart maintenance
        self.maintain_cart_state()
        
        # Bind cleanup to window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    
    def maintain_cart_state(self):
        """Periodically check and maintain cart state"""
        with self.cart_lock:
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_timeout:
                # Only clear if we have items and timeout has passed
                if self.cart_state:
                    self.cart_state = []
                    self.root.after(0, self.display_cart_items)
                    self.root.after(0, self.update_total)
        
        # Schedule next check
        self.root.after(500, self.maintain_cart_state)


    def setup_app_config(self):
        """Initialize app configuration"""
        # App settings
        self.primary_bg = '#be1d37'
        self.secondary_bg = '#f2f2f2'
        self.text_color = '#ffffff'
        self.accent_color = '#f7b2ab'
        self.sidebar_width = 240
        self.main_camera_update_duration = 33  # ~30 FPS
        self.secondary_camera_update_duration = 33
        self.main_camera_index = 0
        self.secondary_camera_index = 1
        
        # Initialize cart
        self.cart = []
        
        # Set window properties
        self.root.title('Tasty Fresh POS System')
        self.root.configure(bg=self.primary_bg)
        
        # Load app icon
        try:
            icon = tk.PhotoImage(file='./Tasty_Fresh_Icon.png')
            self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Could not load icon: {e}")

    def setup_ui(self):
        """Initialize UI components"""
        # Title Frame
        self.setup_title_frame()
        
        # Main Layout
        self.setup_main_layout()
        
        # Cart
        self.setup_cart()
        
        # Status Bar
        self.setup_status_bar()

    def setup_title_frame(self):
        """Setup title frame and logo"""
        self.title_frame = tk.Frame(self.root, bg=self.secondary_bg)
        self.title_frame.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        try:
            title_image = Image.open('./Tasty_Fresh_Title.png')
            self.title_photo = ImageTk.PhotoImage(title_image)
            self.title_label = tk.Label(
                self.title_frame, 
                image=self.title_photo, 
                bg=self.secondary_bg,
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
                bg=self.secondary_bg,
                fg=self.primary_bg
            ).pack()

    def setup_main_layout(self):
        """Setup main layout with cameras"""
        # Main camera frame
        self.main_camera_frame = tk.Frame(
            self.root,
            width=640,
            height=480,
            bg='black',
            bd=2,
            relief='solid'
        )
        self.main_camera_frame.grid(row=1, column=0, padx=10, pady=10)
        
        # Main camera label
        self.main_camera_label = tk.Label(self.main_camera_frame)
        self.main_camera_label.grid(row=0, column=0)
        
        # Sidebar frame
        self.sidebar_frame = tk.Frame(
            self.root,
            width=self.sidebar_width,
            bg=self.primary_bg
        )
        self.sidebar_frame.grid(row=1, column=1, sticky='n', padx=10, pady=10)
        
        # Secondary camera frame
        cam_width = self.sidebar_width - 20
        cam_height = int(cam_width * 3/4)
        self.secondary_camera_frame = tk.Frame(
            self.sidebar_frame,
            width=cam_width,
            height=cam_height,
            bg='gray',
            bd=2,
            relief='solid'
        )
        self.secondary_camera_frame.grid(row=0, column=0, padx=10, pady=5)
        self.secondary_camera_frame.grid_propagate(False)
        
        # Secondary camera label
        self.secondary_camera_label = tk.Label(self.secondary_camera_frame)
        self.secondary_camera_label.grid(row=0, column=0)

    def setup_cart(self):
        """Setup cart display"""
        # Cart frame
        self.cart_frame = tk.Frame(self.sidebar_frame, bg=self.secondary_bg)
        self.cart_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        self.cart_frame.grid_columnconfigure(0, weight=1)
        
        # Scrollable cart
        self.cart_canvas = tk.Canvas(
            self.cart_frame,
            width=self.sidebar_width-25,
            height=250,
            bg=self.secondary_bg,
            bd=0,
            highlightthickness=0
        )
        self.cart_scrollbar = ttk.Scrollbar(
            self.cart_frame,
            orient='vertical',
            command=self.cart_canvas.yview
        )
        
        # Cart items frame
        self.cart_items_frame = tk.Frame(self.cart_canvas, bg=self.secondary_bg)
        self.cart_canvas.create_window(
            (0, 0),
            window=self.cart_items_frame,
            anchor='nw',
            width=self.sidebar_width-45
        )
        
        # Configure scrolling
        self.cart_canvas.configure(yscrollcommand=self.cart_scrollbar.set)
        self.cart_canvas.grid(row=0, column=0, sticky='nsew')
        self.cart_scrollbar.grid(row=0, column=1, sticky='ns')
        self.cart_items_frame.bind('<Configure>', self.update_scroll_region)
        
        # Total label
        self.total_label = tk.Label(
            self.sidebar_frame,
            text='Total: $0.00',
            font=('Helvetica', 14, 'bold'),
            bg=self.primary_bg,
            fg=self.text_color
        )
        self.total_label.grid(row=2, column=0, pady=10)

    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = tk.Label(
            self.root,
            text="System Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg=self.secondary_bg,
            fg=self.primary_bg
        )
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky='ew')

    def setup_cameras(self):
        """Initialize video captures"""
        try:
            self.cap_main = VideoCapture(self.main_camera_index)
            self.cap_secondary = VideoCapture(self.secondary_camera_index)
        except Exception as e:
            print(f"Error initializing cameras: {e}")
            self.status_bar.config(text="Error: Could not initialize cameras")

    def _detection_worker(self):
        """Worker thread for continuous detection processing"""
        while self.detection_running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Run detection using backend
                items = detection_backend.detect(img)
                
                # Update detection results
                with self.cart_lock:
                    self.cart = items
                    self.last_detection_time = time.time()
                    if not self.detection_queue.full():
                        self.detection_queue.put((frame, items))
                
                with self.detection_lock:
                    if items:
                        self.last_detections = items
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
                self.status_bar.config(text=f"Detection error: {str(e)[:50]}...")
                continue

    def update_main_camera(self):
        try:
            frame = self.cap_main.read()
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                display_frame = frame.copy()
                
                # Draw the latest bounding boxes onto the frame
                current_time = time.time()
                 
                with self.detection_lock:
                    #print("Time: ", current_time - self.last_detection_time)
                    if current_time - self.last_detection_time <= self.detection_timeout:
                        for item in self.last_detections:
                            x1, y1, x2, y2 = map(int, item['bbox'])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                display_frame,
                                item['item'],
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )
                            #print("Drew Box")
                    else:
                        # Clear detections if timeout has passed
                        self.last_detections = []
                
                # Queue frame for detection
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                
                # Update camera feed
                img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.main_camera_label.imgtk = imgtk
                self.main_camera_label.configure(image=imgtk)
                
        except Exception as e:
            print(f"Error updating main camera: {e}")
            self.status_bar.config(text="Error updating main camera")
            
        self.root.after(self.main_camera_update_duration, self.update_main_camera)


    def update_secondary_camera(self):
        """Update secondary camera feed"""
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
            print(f"Error updating secondary camera: {e}")
            
        self.root.after(self.secondary_camera_update_duration, self.update_secondary_camera)

    def display_cart_items(self):
        """Display cart items with thread safety and update optimization"""
        with self.cart_lock:
            # Check if cart items have changed
            if self.cart_state == self.displayed_items:
                return
                
            # Update displayed items
            self.displayed_items = self.cart_state.copy()
            
            # Clear existing items in the frame
            for widget in self.cart_items_frame.winfo_children():
                widget.destroy()

            # Add headers
            tk.Label(
                self.cart_items_frame,
                text='Item',
                font=('Helvetica', 12, 'bold'),
                bg=self.secondary_bg,
                fg=self.primary_bg,
                anchor='w'
            ).grid(row=0, column=0, padx=(10, 5), pady=(5, 10), sticky='w')

            tk.Label(
                self.cart_items_frame,
                text='Price',
                font=('Helvetica', 12, 'bold'),
                bg=self.secondary_bg,
                fg=self.primary_bg,
                anchor='e'
            ).grid(row=0, column=1, padx=(5, 10), pady=(5, 10), sticky='e')

            # Add items
            for idx, item in enumerate(self.displayed_items, start=1):
                item_name = item['item']
                item_price = float(item['price']) if 'price' in item else 0.0

                # Create a frame for the item
                item_frame = tk.Frame(self.cart_items_frame, bg=self.secondary_bg)
                item_frame.grid(row=idx, column=0, columnspan=2, sticky='ew')

                tk.Label(
                    item_frame,
                    text=item_name,
                    font=('Helvetica', 11),
                    bg=self.secondary_bg,
                    fg=self.primary_bg,
                    anchor='w'
                ).pack(side='left', padx=(10, 5), pady=2)

                tk.Label(
                    item_frame,
                    text=f'${item_price:.2f}',
                    font=('Helvetica', 11),
                    bg=self.secondary_bg,
                    fg=self.primary_bg,
                    anchor='e'
                ).pack(side='right', padx=(5, 10), pady=2)

            # Update grid configuration
            self.cart_items_frame.grid_columnconfigure(0, weight=1)
            
    def update_total(self):
        """Update total with thread safety"""
        with self.cart_lock:
            total = sum(float(item['price']) for item in self.cart)
            self.total_label.config(text=f'Total: ${total:.2f}')

    def update_scroll_region(self, event=None):
        """Update the scroll region of the cart canvas"""
        self.cart_canvas.configure(scrollregion=self.cart_canvas.bbox('all'))

    def schedule_camera_updates(self):
        """Start camera update loops"""
        self.update_main_camera()
        self.update_secondary_camera()

    def on_closing(self):
        """Clean up resources before closing"""
        self.detection_running = False
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=1.0)
        
        if hasattr(self, 'cap_main'):
            self.cap_main.stop()
        
        if hasattr(self, 'cap_secondary'):
            self.cap_secondary.stop()
        
        self.root.destroy()

def main():
    """Main entry point for the application"""
    root = tk.Tk()
    
    # Set minimum window size
    root.minsize(900, 700)
    
    # Center window on screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 900
    window_height = 700
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    app = POSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()