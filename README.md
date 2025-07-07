# 📊 Sentiment Intensity Analysis of Google Reviews on EduTech Companies in Kochi

This project analyzes the sentiment intensity of Google Reviews from EduTech companies based in Kochi. The solution includes review scraping, NLP-based sentiment classification, readability scoring, and a deployed predictive model, all built using Python and CI/CD integration.

---

## 📌 Project Objective

To analyze public perception of physical EduTech training centers in Kochi by extracting and processing Google Reviews, followed by sentiment intensity analysis using both text-based and rating-based sentiment scores.

---

## 🛠️ Tools & Technologies Used

* **Python** – Data processing, NLP, sentiment scoring
* **Pandas, TextBlob, Scikit-learn, TextStat**
* **Selenium & BeautifulSoup** – Web scraping
* **Power BI** – Visualization and reporting
* **GitHub Actions** – CI/CD pipeline
* **.pkl Model Deployment** – Trained model saved and reused

---

## 🧐 Key Features

* ✅ Google Review Scraping with dynamic scrolling and custom keyword filtering
* ✅ Stack-of-Learning keyword extraction (Python, ML, Data Science, etc.)
* ✅ Sentiment Classification based on review text
* ✅ Readability and linguistic metrics: FOG index, complex words, syllable count
* ✅ Predictive model (ML-based) deployed with reusable `.pkl` files
* ✅ Fully automated pipeline with GitHub Actions
* ✅ CSV output compatible with Power BI dashboarding

---

## 📂 Project Structure

```
├── text_analysis.py                # Final analysis script
├── sentiment_model.pkl             # Trained sentiment classification model
├── vectorizer.pkl                  # TF-IDF/CountVectorizer
├── predict.py / inference.py       # Predict review sentiment
├── requirements.txt                # Python dependencies
├── reviews_with_sentiment.csv      # Output dataset
├── README.md
├── .github/workflows/              # CI/CD setup
└── tests/                          # Unit tests
```

---

## 🚀 How to Run

1. Clone the repo

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the prediction script:

   ```bash
   python predict.py
   ```

4. OR run analysis script with:

   ```bash
   python text_analysis.py
   ```

---

## 📈 Output

* **Final output:** Excel/CSV file with:

  * Review content
  * Rating-based sentiment
  * Text-based sentiment
  * Polarity, subjectivity, FOG index, etc.

* **Power BI Dashboard**:

  * Multi-page dynamic insights on sentiment, stack trends, and temporal analysis

---

## 📬 Contact

**Abel Jacob Varghese**
📧 [abeljacobvarghese@gmail.com](mailto:abeljacobvarghese@gmail.com)
🌐 [LinkedIn Profile](https://linkedin.com/in/abel-jacob-varghese)
