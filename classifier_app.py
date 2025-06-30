import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from keras.models import load_model
from PIL import Image, ImageOps, ImageTk
import numpy as np
import os

class VehicleClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Vehicle AI Classifier")
        self.root.geometry("1000x900")  # Increased size for larger preview
        self.root.configure(bg='#2c3e50')
        self.root.resizable(False, False)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        
        # Initialize model and labels
        self.model = None
        self.class_names = []
        self.load_model_and_labels()
        
        # Create GUI elements
        self.create_widgets()
        
    def load_model_and_labels(self):
        try:
            # Load the model
            if os.path.exists("keras_Model.h5"):
                self.model = load_model("keras_Model.h5", compile=False)
                print("Model loaded successfully!")
            else:
                messagebox.showerror("Error", "keras_Model.h5 not found!")
                return
                
            # Load the labels
            if os.path.exists("labels.txt"):
                self.class_names = open("labels.txt", "r").readlines()
                print("Labels loaded successfully!")
            else:
                messagebox.showerror("Error", "labels.txt not found!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or labels: {str(e)}")
    
    def create_widgets(self):
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        # Header section
        header_frame = tk.Frame(main_frame, bg='#2c3e50')
        header_frame.pack(fill='x', pady=(0, 20))
        
        # Title with icon
        title_label = tk.Label(header_frame, text="🚗 Vehicle AI Classifier", 
                              font=("Segoe UI", 28, "bold"), 
                              bg='#2c3e50', fg='#ecf0f1')
        title_label.pack()
        
        # Subtitle
        subtitle_label = tk.Label(header_frame, text="Advanced AI-powered Car vs Truck Classification", 
                                 font=("Segoe UI", 12), 
                                 bg='#2c3e50', fg='#bdc3c7')
        subtitle_label.pack(pady=(5, 0))
        
        # Status indicator
        self.status_frame = tk.Frame(header_frame, bg='#2c3e50')
        self.status_frame.pack(pady=(10, 0))
        
        status_text = "✅ Model Ready" if self.model else "❌ Model Not Loaded"
        status_color = "#27ae60" if self.model else "#e74c3c"
        self.status_label = tk.Label(self.status_frame, text=status_text,
                                    font=("Segoe UI", 10, "bold"),
                                    bg='#2c3e50', fg=status_color)
        self.status_label.pack()
        
        # Upload section
        upload_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        upload_frame.pack(fill='x', pady=(0, 15))
        
        upload_inner = tk.Frame(upload_frame, bg='#34495e')
        upload_inner.pack(padx=20, pady=15)
        
        upload_title = tk.Label(upload_inner, text="📁 Select Image", 
                               font=("Segoe UI", 14, "bold"),
                               bg='#34495e', fg='#ecf0f1')
        upload_title.pack(pady=(0, 10))
        
        # Styled upload button
        self.upload_btn = tk.Button(upload_inner, text="🖼️ Choose Image File", 
                                   command=self.upload_image,
                                   font=("Segoe UI", 12, "bold"),
                                   bg='#3498db', fg='white',
                                   activebackground='#2980b9',
                                   activeforeground='white',
                                   width=20, height=2,
                                   relief='flat',
                                   cursor='hand2')
        self.upload_btn.pack()
        
        # Image display section - MUCH LARGER
        image_section = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        image_section.pack(fill='both', expand=True, pady=(0, 15))
        
        image_header = tk.Frame(image_section, bg='#34495e')
        image_header.pack(fill='x', padx=20, pady=(10, 0))
        
        image_title = tk.Label(image_header, text="🖼️ Image Preview", 
                              font=("Segoe UI", 14, "bold"),
                              bg='#34495e', fg='#ecf0f1')
        image_title.pack()
        
        # Image display frame - larger container
        self.image_frame = tk.Frame(image_section, bg='#34495e')
        self.image_frame.pack(pady=15, padx=20, fill='both', expand=True)
        
        # Create a canvas for better image display control
        self.image_canvas = tk.Canvas(self.image_frame, 
                                     bg='#2c3e50', 
                                     highlightthickness=0,
                                     relief='sunken', 
                                     bd=2)
        self.image_canvas.pack(fill='both', expand=True)
        
        # Initial text on canvas
        self.image_canvas.create_text(400, 200, 
                                     text="No image selected\n\nClick 'Choose Image File' to get started",
                                     fill='#7f8c8d',
                                     font=("Segoe UI", 12),
                                     justify='center')
        
        # Results section
        results_section = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        results_section.pack(fill='x')
        
        results_inner = tk.Frame(results_section, bg='#34495e')
        results_inner.pack(padx=20, pady=20)
        
        results_title = tk.Label(results_inner, text="🎯 Classification Results", 
                                font=("Segoe UI", 14, "bold"),
                                bg='#34495e', fg='#ecf0f1')
        results_title.pack(pady=(0, 15))
        
        # Results display
        self.results_frame = tk.Frame(results_inner, bg='#2c3e50', relief='sunken', bd=2)
        self.results_frame.pack(fill='x', pady=5)
        
        # Prediction label
        self.prediction_label = tk.Label(self.results_frame, text="Awaiting classification...", 
                                        font=("Segoe UI", 16, "bold"),
                                        bg='#2c3e50', fg='#ecf0f1')
        self.prediction_label.pack(pady=15)
        
        # Confidence label
        self.confidence_label = tk.Label(self.results_frame, text="", 
                                        font=("Segoe UI", 12),
                                        bg='#2c3e50')
        self.confidence_label.pack(pady=(0, 15))
        
        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(results_inner, mode='indeterminate', length=300)
        
    def upload_image(self):
        if self.model is None:
            messagebox.showerror("❌ Error", "Model not loaded!")
            return
            
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.classify_image(file_path)
    
    def classify_image(self, image_path):
        try:
            # Show progress
            self.progress.pack(pady=10)
            self.progress.start(10)
            self.root.update()
            
            # Display the selected image FIRST
            self.display_image(image_path)
            
            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            
            # Load and process the image
            image = Image.open(image_path).convert("RGB")
            
            # Resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            
            # Turn the image into a numpy array
            image_array = np.asarray(image)
            
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            
            # Load the image into the array
            data[0] = normalized_image_array
            
            # Predicts the model
            prediction = self.model.predict(data, verbose=0)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]
            
            # Hide progress
            self.progress.stop()
            self.progress.pack_forget()
            
            # Display results
            self.display_results(class_name, confidence_score)
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("❌ Error", f"Failed to classify image: {str(e)}")
    
    def display_image(self, image_path):
        try:
            # Clear the canvas
            self.image_canvas.delete("all")
            
            # Open the original image
            original_image = Image.open(image_path)
            
            # Get canvas dimensions
            self.image_canvas.update()
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # If canvas dimensions are not ready, use default
            if canvas_width <= 1:
                canvas_width = 750
            if canvas_height <= 1:
                canvas_height = 400
            
            # Add some padding
            max_width = canvas_width - 20
            max_height = canvas_height - 20
            
            # Get original dimensions
            orig_width, orig_height = original_image.size
            
            # Calculate the scaling factor to fit the image without cropping
            width_ratio = max_width / orig_width
            height_ratio = max_height / orig_height
            scale_factor = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)
            
            # Resize the image maintaining aspect ratio
            resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(resized_image)
            
            # Calculate position to center the image
            x = canvas_width // 2
            y = canvas_height // 2
            
            # Display image on canvas
            self.image_canvas.create_image(x, y, image=self.photo, anchor='center')
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.image_canvas.delete("all")
            self.image_canvas.create_text(400, 200, 
                                         text=f"Error loading image:\n{str(e)}",
                                         fill='#e74c3c',
                                         font=("Segoe UI", 12),
                                         justify='center')
    
    def display_results(self, class_name, confidence_score):
        # Extract class name (remove index number like "0 " or "1 ")
        clean_class_name = class_name[2:].strip()
        
        # Get appropriate emoji and color
        if clean_class_name.lower() == "cars":
            emoji = "🚗"
            result_color = "#3498db"
        else:  # Truck
            emoji = "🚛"
            result_color = "#e67e22"
        
        # Update prediction label
        self.prediction_label.configure(text=f"{emoji} Detected: {clean_class_name}",
                                       fg=result_color)
        
        # Update confidence label with color coding
        confidence_percent = confidence_score * 100
        confidence_text = f"Confidence: {confidence_percent:.1f}%"
        
        # Color code based on confidence
        if confidence_score >= 0.8:
            conf_color = "#27ae60"  # Green
            conf_emoji = "🟢"
        elif confidence_score >= 0.6:
            conf_color = "#f39c12"  # Orange
            conf_emoji = "🟡"
        else:
            conf_color = "#e74c3c"  # Red
            conf_emoji = "🔴"
            
        self.confidence_label.configure(text=f"{conf_emoji} {confidence_text}", 
                                       fg=conf_color,
                                       font=("Segoe UI", 12, "bold"))
        
        # Also print to console
        print(f"Class: {clean_class_name}")
        print(f"Confidence Score: {confidence_score:.4f}")

def main():
    # Create the main window
    root = tk.Tk()
    
    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1000 // 2)
    y = (root.winfo_screenheight() // 2) - (900 // 2)
    root.geometry(f"1000x900+{x}+{y}")
    
    # Create the application
    app = VehicleClassifier(root)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()