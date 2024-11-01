import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import json
import queue
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from backend import detect

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only the most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        if not self.q.empty():
            return self.q.get()
        return None

class POSApp:
    def __init__(self, root):
        self.root = root

        # Initialise async loop for camera and detection
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # App title
        self.root.title('POS System')

        # App icon
        icon = tk.PhotoImage(file='./Tasty_Fresh_Icon.png')
        self.root.iconphoto(True, icon)

        # Colours and config
        self.primary_bg = '#be1d37'
        self.secondary_bg = '#f2f2f2'
        self.text_color = '#ffffff'
        self.accent_color = '#f7b2ab'
        self.main_camera_index = 0
        self.secondary_camera_index = 6
        self.sidebar_width = 240
        self.main_camera_update_duration = 500
        self.secondary_camera_update_duration = 100

        # Set background
        self.root.configure(bg=self.primary_bg)

        # Title component
        self.title_frame = tk.Frame(root, bg=self.secondary_bg)
        self.title_frame.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        title_image = Image.open('./Tasty_Fresh_Title.png')
        self.title_photo = ImageTk.PhotoImage(title_image)
        self.title_label = tk.Label(self.title_frame, image=self.title_photo, bg=self.secondary_bg, bd=0, highlightthickness=0)
        self.title_label.pack()

        # Main camera component
        self.main_camera_frame = tk.Frame(root, width=640, height=480, bg='black', bd=2, relief='solid')
        self.main_camera_frame.grid(row=1, column=0, padx=10, pady=10)

        # Sidebar component
        self.sidebar_frame = tk.Frame(root, width=self.sidebar_width, bg=self.primary_bg)
        self.sidebar_frame.grid(row=1, column=1, sticky='n', padx=10, pady=10)

        # Calculate secondary camera dimensions maintaining 4:3 aspect ratio
        cam_width = self.sidebar_width - 20  # Account for padding
        cam_height = int(cam_width * 3/4)  # Maintain 4:3 aspect ratio

        # Secondary camera component
        self.secondary_camera_frame = tk.Frame(self.sidebar_frame, width=cam_width, height=cam_height, bg='gray', bd=2, relief='solid')
        self.secondary_camera_frame.grid(row=0, column=0, padx=10, pady=5)
        self.secondary_camera_frame.grid_propagate(False)

        # Cart component
        self.cart_frame = tk.Frame(self.sidebar_frame, bg=self.secondary_bg)
        self.cart_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        self.cart_frame.grid_columnconfigure(0, weight=1)

        # Allow cart to be scrollable if item list is too long
        self.cart_canvas = tk.Canvas(self.cart_frame, width=cam_width-5, height=250, bg=self.secondary_bg, bd=0, highlightthickness=0)
        self.cart_scrollbar = tk.Scrollbar(self.cart_frame, orient='vertical', command=self.cart_canvas.yview)
        self.cart_items_frame = tk.Frame(self.cart_canvas, bg=self.secondary_bg)
        self.cart_canvas.create_window((0, 0), window=self.cart_items_frame, anchor='nw', width=cam_width-25)
        self.cart_canvas.configure(yscrollcommand=self.cart_scrollbar.set)
        self.cart_canvas.grid(row=0, column=0, sticky='nsew')
        self.cart_scrollbar.grid(row=0, column=1, sticky='ns')
        self.cart_items_frame.bind('<Configure>', self.update_scroll_region)
        
        # Total cost component
        self.total_label = tk.Label(self.sidebar_frame, text='', font=('Helvetica', 14, 'bold'), 
                                  bg=self.primary_bg, fg=self.text_color)
        self.total_label.grid(row=2, column=0, pady=10)

        # Initialize camera labels
        self.main_camera_label = tk.Label(self.main_camera_frame)
        self.main_camera_label.grid(row=0, column=0)

        self.secondary_camera_label = tk.Label(self.secondary_camera_frame)
        self.secondary_camera_label.grid(row=0, column=0)

        # Initialize video feeds
        self.cap_main = VideoCapture(self.main_camera_index)
        self.cap_secondary = VideoCapture(self.secondary_camera_index)

        # Initialize ThreadPoolExecutor for detection
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.cart = []

        # Start async camera and detection updates
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()

        # Schedule the camera updates using Tkinter's after method
        self.schedule_camera_updates()

    def _run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def schedule_camera_updates(self):
        self.root.after(0, self.update_main_camera)
        self.root.after(0, self.update_secondary_camera)

    def update_main_camera(self):
        async def process_frame():
            # Get camera input
            frame = self.cap_main.read()
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Detect items within the image
                start_time = time.time()
                items = await self.loop.run_in_executor(self.executor, detect, img)
                end_time = time.time()
                detection_time = end_time - start_time
                print(f"Detection Time: {detection_time:.4f} seconds")

                # Store data
                self.cart = items
                self.last_image = img

                # Annotate image
                for item in items:
                    name = item['item']
                    bbox = item['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Update UI in main thread
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.root.after(0, self._update_main_camera_ui, imgtk)

        asyncio.run_coroutine_threadsafe(process_frame(), self.loop)
        self.root.after(self.main_camera_update_duration, self.update_main_camera)

    def _update_main_camera_ui(self, imgtk):
        self.main_camera_label.imgtk = imgtk
        self.main_camera_label.configure(image=imgtk)
        self.display_cart_items()
        self.update_total()

    def update_secondary_camera(self):
        # Get camera input
        frame = self.cap_secondary.read()
        if frame is not None:
            cam_width = self.sidebar_width - 20
            cam_height = int(cam_width * 3/4)
            frame = cv2.resize(frame, (cam_width, cam_height))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.secondary_camera_label.imgtk = imgtk
            self.secondary_camera_label.configure(image=imgtk)

        self.root.after(self.secondary_camera_update_duration, self.update_secondary_camera)

    def display_cart_items(self):
        # Configure columns in cart items frame
        self.cart_items_frame.grid_columnconfigure(0, weight=1)
        self.cart_items_frame.grid_columnconfigure(1, weight=0)

        # Add header (only once)
        if not self.cart_items_frame.winfo_children():
            tk.Label(self.cart_items_frame, text='Item', font=('Helvetica', 12, 'bold'),
                     bg=self.secondary_bg, fg=self.primary_bg, anchor='w'
                     ).grid(row=0, column=0, padx=(10, 5), pady=(5, 10), sticky='w')

            tk.Label(self.cart_items_frame, text='Price', font=('Helvetica', 12, 'bold'),
                     bg=self.secondary_bg, fg=self.primary_bg, anchor='e'
                     ).grid(row=0, column=1, padx=(5, 10), pady=(5, 10), sticky='e')

        # Clear existing items but maintain header
        for widget in self.cart_items_frame.winfo_children()[2:]:
            widget.destroy()
        
        # Add items to the cart
        for idx, item in enumerate(self.cart, start=1):
            item_name = item['item']
            item_price = float(item['price']) if 'price' in item else 0.0  # Handle potential missing price

            # Item name (left-aligned)
            tk.Label(self.cart_items_frame, text=item_name, font=('Helvetica', 11),
                    bg=self.secondary_bg, fg=self.primary_bg, anchor='w'
                    ).grid(row=idx, column=0, padx=(10, 5), pady=2, sticky='w')

            # Price (right-aligned)
            tk.Label(self.cart_items_frame, text=f'${item_price:.2f}', font=('Helvetica', 11),
                    bg=self.secondary_bg, fg=self.primary_bg, anchor='e'
                    ).grid(row=idx, column=1, padx=(5, 10), pady=2, sticky='e')

    def update_total(self):
        # Calculate total from the items in the cart
        total = sum(float(item['price']) for item in self.cart)
        self.total_label.config(text=f'Total: ${total:.2f}')

    def update_scroll_region(self, event=None):
        self.cart_canvas.configure(scrollregion=self.cart_canvas.bbox('all'))
        content_height = self.cart_canvas.bbox('all')[3]
        if content_height <= 250:
            self.cart_scrollbar.grid_remove()
        else:
            self.cart_scrollbar.grid()

if __name__ == '__main__':
    root = tk.Tk()
    app = POSApp(root)
    root.mainloop()