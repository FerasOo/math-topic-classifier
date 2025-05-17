# Math Topic Classification Frontend

A modern, single-page web application for classifying mathematical questions into different topics using an Azure ML deployed model.

## ✨ Features

- **Modern UI Design**: Clean, intuitive interface with animations and visual feedback
- **Single-Page Application**: Results appear on the same page without redirects
- **Random Question Generator**: One-click button to insert sample math questions
- **Responsive Design**: Works beautifully on mobile, tablet and desktop
- **Real-Time Feedback**: Visual confidence meter with color-coding
- **Topic Icons**: Visual representations for each math topic

## 🚀 Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the FastAPI application:
   ```
   cd frontend
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## 🔍 How It Works

1. User enters a math question or generates a random one with the dice button
2. The application sends an AJAX request to the backend
3. The backend forwards the question to an Azure ML endpoint
4. The ML model classifies the question into one of 8 math topics
5. Results appear dynamically on the same page with a confidence score
6. A color-coded confidence bar visualizes the model's certainty

## 🛠️ Implementation Details

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: FastAPI with asynchronous endpoints
- **API Integration**: AJAX calls to Azure ML endpoint
- **Authentication**: Bearer token for secure access to Azure ML endpoint
- **Design**: Modern UI with animations, gradients and responsive layout

## 📊 Topics Covered

- Algebra
- Geometry and Trigonometry
- Calculus and Analysis
- Probability and Statistics
- Number Theory
- Combinatorics and Discrete Math
- Linear Algebra
- Abstract Algebra and Topology

## 🧩 Features for Future Improvement

- User accounts to save question history
- Exporting results as PDF or CSV
- Explanation of why a question belongs to a specific topic
- Related questions based on the current classification
- Dark mode toggle 