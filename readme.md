Diagnosis X-Ray and Brain MRI Detector

This web application lets you upload chest X-rays or brain MRIs and receive predictions:
- Whether a patient has pneumonia or not
- Type of brain tumor (glioma, meningioma, pituitary) or no tumor

Technologies used:
- Flask
- TensorFlow and Keras
- scikit-learn
- HTML, CSS, and JavaScript

Features:
- Upload medical images for prediction
- Get simple diagnosis results
- Clear explanation of predictions in user-friendly language

Project Structure:
your-project/
├── app.py
├── models/
│   ├── final_pneumonia_model.keras
│   └── best_brain_tumor_model.keras
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
├── Dockerfile
└── README.md

How to Run Locally:
1. Clone the repository:
   git clone https://github.com/yourusername/diagnosis-ai.git
   cd diagnosis-ai

2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  (on Windows: venv\Scripts\activate)
   pip install -r requirements.txt

3. Place your trained model files in the models directory:
   models/
   ├── final_pneumonia_model.keras
   └── best_brain_tumor_model.keras

4. Run the app:
   python app.py

5. Open your browser at:
   http://127.0.0.1:5000

Running with Docker:
1. Build the Docker image:
   docker build -t diagnosis-ai .

2. Run the Docker container:
   docker run -p 5000:5000 diagnosis-ai

Then visit http://localhost:5000 in your browser.

Notes:
- Ensure models are named and saved correctly in the /models folder
- Compatible with Python 3.10 and above
