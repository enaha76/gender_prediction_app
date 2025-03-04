# Name Gender Predictor

A Streamlit web application that predicts the gender associated with a name using machine learning models (Logistic Regression and Naive Bayes) built with PySpark ML.

## Features

- Predicts gender based on input names using two different machine learning models
- Interactive UI with tabs for different models and data exploration
- Data visualization with charts and metrics
- Model comparison functionality
- Clean and intuitive user interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/name-gender-predictor.git
cd name-gender-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Make sure Java is installed for PySpark:
```bash
# On Ubuntu/Debian
sudo apt-get install default-jdk

# On macOS
brew install openjdk@11

# On Windows
# Download and install from https://www.oracle.com/java/technologies/javase-jdk11-downloads.html
```

## Usage

1. Place your training data file (names-mr.csv) in the `data/` directory.

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and go to http://localhost:8501

4. In the app:
   - Click "Load Data & Train Models" in the sidebar
   - Once the models are trained, use the tabs to switch between data overview and different models
   - Enter a name in the text input field to get gender predictions

## Project Structure

```
gender_prediction_app/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── README.md               # Documentation
│
├── data/
│   └── names-mr.csv        # Training data
│
├── models/
│   ├── __init__.py
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── ml_models.py        # Machine learning models
│   └── saved/              # Directory for saved models
│
└── utils/
    ├── __init__.py
    └── helpers.py          # Helper functions
```

## Deployment

### Deploy to Streamlit Sharing

1. Push your code to a GitHub repository

2. Go to [Streamlit Sharing](https://share.streamlit.io/)

3. Sign in with GitHub and click "New app"

4. Select your repository, branch, and main file path (app.py)

5. Click "Deploy"

### Deploy to Heroku

1. Create a new file named `Procfile` with the following content:
```
web: streamlit run app.py --server.port $PORT
```

2. Create a `setup.sh` file:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Install Heroku CLI and login:
```bash
heroku login
```

4. Create a Heroku app:
```bash
heroku create your-app-name
```

5. Set Java buildpack for PySpark:
```bash
heroku buildpacks:add heroku/jvm
heroku buildpacks:add heroku/python
```

6. Deploy to Heroku:
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The application uses PySpark ML for machine learning models
- UI built with Streamlit
- Visualization using Matplotlib and Seaborn