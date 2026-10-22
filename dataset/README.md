# Dataset Directory for FakeLens

Please place the Kaggle Fake and Real News dataset CSV files in this directory:

- `Fake.csv` (Contains fake news articles)
- `True.csv` (Contains real news articles)

### Source:
You can download these files from Kaggle:
- **Dataset Link**: [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Alternatively from the ISOT Research Lab Fake News Dataset repository.

### Expected Schema:
Both CSV files should contain the following columns:
- `title`: Title of the news article
- `text`: Main body text of the article
- `subject`: Category / topic of the news
- `date`: Publication date

### Next Steps:
Once `Fake.csv` and `True.csv` are placed here, run:
```bash
# 1. Run Exploratory Data Analysis & generate plots for your presentation
python eda/eda.py

# 2. Train the TF-IDF + Logistic Regression model, evaluate metrics, and run leakage analysis
python model/train_model.py

# 3. Launch the web application
python backend/app.py
```
