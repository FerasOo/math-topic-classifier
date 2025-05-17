import requests
import json
import random
import os
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="Math Topic Classification")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Azure ML endpoint details
ENDPOINT_URL = "https://math-topic-prediction-endpoint.francecentral.inference.ml.azure.com/score"
API_KEY = os.getenv("API_KEY")
print("API_KEY: ", API_KEY)
API_KEY = "33zdSUzOXnJU1rjTyCV6vh0w8Nez5H7ObC8XpptSJUUlnGlNTcX4JQQJ99BEAAAAAAAAAAAAINFRAZML2cLM"

# Topic mapping
TOPICS = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

# Sample math questions for the random question feature - 3 for each topic
SAMPLE_QUESTIONS = [
    "Solve the quadratic equation x^2 - 5x + 6 = 0.",                                   # Algebra
    "Factor the expression 2x³ - 4x² - 22x + 24.",                                      # Algebra
    "Simplify the expression (3x⁴y²)³ ÷ (9x²y⁵)².",                                     # Algebra
    
    "Find the area of a triangle with vertices at (0,0), (3,4), and (-1,2).",           # Geometry and Trigonometry
    "Calculate the volume of a cone with radius 5 cm and height 12 cm.",                # Geometry and Trigonometry
    "If sin(θ) = 0.6, what is the value of cos(θ)?",                                    # Geometry and Trigonometry
    
    "Find the limit of (1+1/n)^n as n approaches infinity.",                            # Calculus and Analysis
    "Calculate the derivative of f(x) = x^3 - 3x^2 + 2x - 1.",                          # Calculus and Analysis
    "Evaluate the integral of x*ln(x) from x=1 to x=e.",                                # Calculus and Analysis
    
    "If P(A) = 0.3 and P(B) = 0.4 and A and B are independent events, what is P(A and B)?", # Probability and Statistics
    "What is the probability of rolling a sum of 8 with two dice?",                     # Probability and Statistics
    "Calculate the mean and standard deviation of the dataset: 4, 7, 8, 11, 12, 14.",   # Probability and Statistics
    
    "Find all prime numbers p such that p+2 is also prime.",                            # Number Theory
    "Prove that the square root of 3 is irrational.",                                   # Number Theory
    "Find the greatest common divisor of 168 and 180.",                                 # Number Theory
    
    "How many ways can 5 books be arranged on a shelf?",                                # Combinatorics and Discrete Math
    "In how many ways can a committee of 3 be chosen from 9 people?",                   # Combinatorics and Discrete Math
    "Solve the recurrence relation an = an-1 + 2an-2 with a0 = 1 and a1 = 3.",         # Combinatorics and Discrete Math
    
    "For the matrix A = [[1,2,3],[4,5,6],[7,8,9]], find the eigenvalues and eigenvectors.",                           # Linear Algebra
    "Determine if the vectors (1,2,3), (2,3,4), and (3,4,5) are linearly independent.", # Linear Algebra
    "Find the eigenvalues of the matrix [[4,1],[6,-1]].",                               # Linear Algebra
    
    "Prove that the set of continuous functions on [0,1] is a vector space.",           # Abstract Algebra and Topology
    "Show that the set of all rational numbers is dense in the real numbers.",          # Abstract Algebra and Topology
    "Prove that every compact metric space is complete."                                # Abstract Algebra and Topology
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(question: str = Form(...)):
    try:
        # Prepare request to Azure ML endpoint
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input_data": [question]
        }
        
        # Call the Azure ML endpoint
        response = requests.post(ENDPOINT_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the result
        result = response.json()
        
        return {
            "success": True,
            "question": question,
            "prediction": result[0]["label"],
            "confidence": round(result[0]["score"] * 100, 2)
        }
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e)
        }

@app.get("/random-question", response_class=JSONResponse)
async def get_random_question():
    question = random.choice(SAMPLE_QUESTIONS)
    return {"question": question}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 