import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Region growing algorithm
def region_growing(image, seed, threshold=0.1):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    connectivity = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    seed_list = [seed]
    
    while seed_list:
        x, y = seed_list.pop(0)
        if segmented[x, y] == 1:
            continue
        segmented[x, y] = 1
        pixel_value = image[x, y]
        for dx, dy in connectivity:
            xn, yn = x + dx, y + dy
            if 0 <= xn < h and 0 <= yn < w and segmented[xn, yn] == 0:
                if abs(image[xn, yn] - pixel_value) < threshold:
                    seed_list.append((xn, yn))
    return segmented

# Main Application
class KidneySegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kidney Instance Segmentation using Region Growing")
        
        # Variables
        self.image = None
        self.img_path = None
        self.seed_point = None
        self.processed_mask = None
        
        # GUI Components
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.pack()

        # Load Image Button
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Run Region Growing Button
        self.run_button = tk.Button(self.root, text="Run Region Growing", command=self.run_region_growing)
        self.run_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Bind the canvas to mouse click event for seed point selection
        self.canvas.bind("<Button-1>", self.get_seed_point)
        
    # Load Image
    def load_image(self):
        self.img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if self.img_path:
            self.image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)

    # Display image on the canvas
    def display_image(self, img):
        # Convert image for displaying in Tkinter
        img = cv2.resize(img, (500, 500))  # Resize for display purposes
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img)
        
        # Update the canvas image
        self.canvas.imgtk = img_tk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # Get seed point from mouse click
    def get_seed_point(self, event):
        if self.image is None:
            return
        # Get the mouse click coordinates on the image
        x, y = event.x, event.y
        # Convert coordinates to the original image scale (not the resized one)
        h, w = self.image.shape
        scale_x = w / 500
        scale_y = h / 500
        self.seed_point = (int(y * scale_y), int(x * scale_x))  # (row, col)
        print(f"Seed Point Selected: {self.seed_point}")
        
    # Run Region Growing Algorithm
    def run_region_growing(self):
        if self.seed_point and self.image is not None:
            mask = region_growing(self.image, self.seed_point, threshold=0.1)
            self.processed_mask = mask
            self.display_image(mask * 255)  # Display mask
            self.visualize_mask()

    # Visualize Mask using Matplotlib
    def visualize_mask(self):
        if self.image is not None and self.processed_mask is not None:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(self.image, cmap='gray')
            plt.title('Original Image')

            plt.subplot(1, 2, 2)
            plt.imshow(self.processed_mask, cmap='gray')
            plt.title('Segmented Mask (Region Growing)')
            
            plt.show()

# Start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = KidneySegmentationApp(root)
    root.mainloop()
