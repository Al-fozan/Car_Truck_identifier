# 🚗 Vehicle AI Classifier

A simple desktop app that uses AI to tell if an image shows a car or truck.

## What it does

- Upload any image
- AI analyzes it 
- Shows if it's a car 🚗 or truck 🚛
- Displays confidence percentage

## How to install

1. **Download the files**
   ```bash
   git clone https://github.com/yourusername/vehicle-ai-classifier.git
   cd vehicle-ai-classifier
   ```

2. **Install Python packages**
   ```bash
   pip install tensorflow==2.12.1 pillow numpy
   ```

3. **Make sure you have these files:**
   - `classifier_app.py` (the main program)
   - `keras_Model.h5` (the AI model)
   - `labels.txt` (contains: "0 Cars" and "1 Truck")

## How to use

1. **Run the app:**
   ```bash
   python classifier_app.py
   ```

2. **Use it:**
   - Click "Choose Image File"
   - Select any car or truck picture
   - See the result!

## Screenshots

*Add a screenshot of your app here*

## Requirements

- Python 3.8+
- Windows, Mac, or Linux

## Files needed

```
├── classifier_app.py    # Main program
├── keras_Model.h5       # AI model file  
└── labels.txt           # Labels file
```

## Troubleshooting

**"Model not found" error?**
- Make sure `keras_Model.h5` is in the same folder as `classifier_app.py`

**"Labels not found" error?**
- Make sure `labels.txt` is in the same folder and contains:
  ```
  0 Cars
  1 Truck
  ```

## License

Free to use and modify.

---

⭐ Star this project if you like it!