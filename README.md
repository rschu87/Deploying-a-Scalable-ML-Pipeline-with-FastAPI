# Deploying a Machine Learning Model with FastAPI

**GitHub repository:** https://github.com/rschu87/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

WGU D501 Machine Learning DevOps project. A random forest classifier trained on Census data predicts whether income is above or below $50K, and a FastAPI app serves the predictions.

## Project files
* `ml/model.py` and `ml/data.py`: training, inference, save/load, metrics and slice functions
* `train_model.py`: trains the model, saves `model/model.pkl` and `model/encoder.pkl`, and writes `slice_output.txt`
* `test_ml.py`: unit tests (run with `pytest test_ml.py -v`)
* `main.py`: the FastAPI app (GET `/` and POST `/data/`)
* `local_api.py`: sends one GET and one POST to the running API
* `model_card.md`: the model card
* `screenshots/`: `continuous_integration.png`, `unit_test.png`, `local_api.png`

## How to run
```
conda env create -f environment.yml
conda activate fastapi
python train_model.py
pytest test_ml.py -v
uvicorn main:app --reload
python local_api.py   # in a second terminal
```

---

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
