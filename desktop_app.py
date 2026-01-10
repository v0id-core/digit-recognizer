import tkinter as tk
from tkinter import messagebox
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import threading

from model import ConvNet
import train

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer | v0id-core")
        self.root.geometry("400x480")
        self.root.resizable(False, False)

        # State
        self.model_loaded = False
        self.model_path = "mnist_model.pt"
        self.model = ConvNet()

        # --- UI Setup ---
        
        # 1. Canvas
        self.canvas_width = 280
        self.canvas_height = 280
        self.bg_color = "white"
        self.paint_color = "black"
        
        self.canvas = tk.Canvas(
            self.root, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg=self.bg_color, 
            relief="sunken", 
            bd=2
        )
        self.canvas.pack(pady=20)

        # Internal Image (High Quality)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)

        # Events
        self.canvas.bind("<B1-Motion>", self.paint)

        # 2. Status Label
        self.lbl_result = tk.Label(self.root, text="Checking model...", font=("Helvetica", 14))
        self.lbl_result.pack(pady=5)

        # 3. Buttons
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        # Recognize Button (initially disabled)
        self.btn_predict = tk.Button(self.btn_frame, text="Recognize", command=self.predict_digit, width=12, bg="#dddddd", state=tk.DISABLED)
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(self.btn_frame, text="Clear", command=self.clear_canvas, width=10)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.btn_train = tk.Button(self.btn_frame, text="Train Model", command=self.start_training_thread, width=12)
        self.btn_train.pack(side=tk.LEFT, padx=5)

        self.btn_exit = tk.Button(self.root, text="Exit", command=self.root.quit, width=10, fg="red")
        self.btn_exit.pack(pady=10)

        # Try to load model on startup
        self.load_model()

    def load_model(self):
        """Loads model and updates UI state."""
        try:
            # weights_only=True for security
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
            self.model.eval()
            self.model_loaded = True
            
            # Update UI
            self.lbl_result.config(text="Model Ready! Draw a digit.", fg="green")
            self.btn_predict.config(state=tk.NORMAL, bg="#90ee90") # Greenish button
            
        except FileNotFoundError:
            self.model_loaded = False
            self.lbl_result.config(text="Model NOT found.\nPlease click 'Train Model'!", fg="red")
            self.btn_predict.config(state=tk.DISABLED, bg="#dddddd")

    def paint(self, event):
        """Draw with a thicker brush for better recognition."""
        r = 10  # Increased radius (thicker line)
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline=self.paint_color)
        self.draw.ellipse([x1, y1, x2, y2], fill=self.paint_color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        if self.model_loaded:
            self.lbl_result.config(text="Draw a digit...", fg="black")

    def smart_center_image(self, img):
        """
        Auto-crops the digit and centers it in a 28x28 frame.
        This mimics the MNIST dataset structure.
        """
        # 1. Invert (Black background, White digit)
        img = ImageOps.invert(img)
        
        # 2. Get Bounding Box (crop empty space)
        bbox = img.getbbox()
        if bbox is None:
            return None # Image is empty
            
        cropped = img.crop(bbox)
        
        # 3. Resize preserving aspect ratio to fit in 20x20 box (leaving padding)
        cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
        
        # 4. Paste centered into a 28x28 black image
        new_img = Image.new("L", (28, 28), 0) # 0 = Black
        
        # Calculate center position
        x_offset = (28 - cropped.width) // 2
        y_offset = (28 - cropped.height) // 2
        
        new_img.paste(cropped, (x_offset, y_offset))
        return new_img

    def predict_digit(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded!")
            return

        try:
            # Use smart centering
            processed_img = self.smart_center_image(self.image)
            
            if processed_img is None:
                self.lbl_result.config(text="Canvas is empty!", fg="orange")
                return

            # Transform to Tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            img_tensor = transform(processed_img).unsqueeze(0)

            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                prediction = output.argmax(dim=1, keepdim=True).item()
                probability = torch.exp(output).max().item() * 100

            self.lbl_result.config(text=f"Prediction: {prediction} ({probability:.1f}%)", fg="blue")
            
        except Exception as e:
            self.lbl_result.config(text="Error predicting", fg="red")
            print(e)

    def start_training_thread(self):
        self.lbl_result.config(text="Training... (Wait ~30s)", fg="orange")
        self.btn_train.config(state=tk.DISABLED)
        self.btn_predict.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_training_process)
        thread.start()

    def run_training_process(self):
        status = train.run_training()
        self.load_model() # Reload to enable buttons
        self.lbl_result.config(text=status, fg="green")
        self.btn_train.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Training finished!")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()