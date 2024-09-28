import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class KidneySegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kidney Instance Segmentation")
        self.root.geometry("1000x700")

        self.seed_points = []
        self.threshold = tk.DoubleVar(value=0.2)
        self.mask = None
        self.img = None
        self.loaded_mask = None

        # GUI layout
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.threshold_label = tk.Label(self.controls_frame, text="Threshold")
        self.threshold_label.pack()

        self.threshold_entry = tk.Entry(self.controls_frame, textvariable=self.threshold)
        self.threshold_entry.pack()

        self.load_image_btn = tk.Button(self.controls_frame, text="Load Image", command=self.load_image)
        self.load_image_btn.pack(pady=10)

        self.generate_mask_btn = tk.Button(self.controls_frame, text="Generate Mask", command=self.generate_mask)
        self.generate_mask_btn.pack(pady=10)

        self.load_mask_btn = tk.Button(self.controls_frame, text="Load Previous Mask", command=self.load_mask)
        self.load_mask_btn.pack(pady=10)

        self.save_mask_btn = tk.Button(self.controls_frame, text="Save Mask", command=self.save_mask)
        self.save_mask_btn.pack(pady=10)

        self.clear_btn = tk.Button(self.controls_frame, text="Clear Seed Points", command=self.clear_seeds)
        self.clear_btn.pack(pady=10)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('button_press_event', self.onclick)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.img)

    def display_image(self, img):
        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        self.canvas.draw()

    def onclick(self, event):
        if self.img is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.seed_points.append((x, y))
            self.ax.plot(x, y, 'ro')
            self.canvas.draw()

    def generate_mask(self):
        if self.img is not None and self.seed_points:
            self.mask = np.zeros_like(self.img, dtype=np.uint8)
            for seed in self.seed_points:
                self.region_growing(seed)
            self.display_image(self.mask)

    def region_growing(self, seed):
        x, y = seed
        thresh = self.threshold.get()

        # OpenCV region growing
        connectivity = 8
        mask = np.zeros((self.img.shape[0]+2, self.img.shape[1]+2), np.uint8)  # Extra border for floodFill
        cv2.floodFill(self.img, mask, (x, y), 255, loDiff=(thresh*255,), upDiff=(thresh*255,), flags=connectivity)

        self.mask = cv2.bitwise_or(self.mask, mask[1:-1, 1:-1])  # Apply mask ignoring border

    def save_mask(self):
        if self.mask is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, self.mask)
                messagebox.showinfo("Saved", "Mask saved successfully!")

    def load_mask(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png")])
        if file_path:
            self.loaded_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.loaded_mask)
            self.mask = self.loaded_mask.copy()

    def clear_seeds(self):
        self.seed_points.clear()
        if self.img is not None:
            self.display_image(self.img)

if __name__ == "__main__":
    root = tk.Tk()
    app = KidneySegmentationApp(root)
    root.mainloop()
