# DS8001 Course Project – Recommendation Systems  
**Empirical Analysis of Recommendation Algorithms**

## Course Information
- **Course**: DS8001 – Design of Algorithms and Programming for Massive Data  
- **Project Type**: Empirical Analysis  
- **Dataset**: The Movies Dataset (Kaggle)  
  https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

## Team Members
- Devanshu Prajapati  
- Taran Veer Singh  

---

## Project Overview
This project compares multiple recommendation algorithms using a real-world movie dataset.  
Each algorithm is implemented in a separate Python file and can be executed independently.

Implemented approaches:
1. Content-Based Filtering
2. Collaborative Filtering
3. Popularity-Based Recommendation
4. Hybrid Recommendation
5. Latent Factor Model (SVD)

---

## Repository Structure
- `archive/` — folder containing all datasets downloaded from Kaggle  (as data is large enough you have to download it).
- `1.1.Content_based.py`  
- `1.2.Content_based_Recommender.py`  
- `Collaborative_Filtering.py`  
- `Popularity_Based.py`  
- `Hybrid_Based_Recommender.py`  
- `5.1.Latent_Model.py`  
- `5.2.Latent_Recommender.py`  
- `README.md`  

---

## Dataset
The project uses **The Movies Dataset** from Kaggle.
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Required files inside the `archive/` folder:
- movies_metadata.csv  
- credits.csv  
- keywords.csv  
- links.csv  
- links_small.csv  
- ratings.csv  
- ratings_small.csv  

**Dataset Setup**
- Download the dataset ZIP from Kaggle  
- Extract the ZIP file  
- Rename the extracted folder to **archive**  
- Place the `archive` folder in the same directory as all Python files  

All scripts access data using paths such as:
`archive/movies_metadata.csv`

---

## Installation and Requirements

### Required Libraries
Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk surprise
```
---

# Content-Based Recommendation System

## How to Run

**Step 1:**  
Run `1.1.Content_based.py`.  
This script performs preprocessing and generates similarity models.  
Wait until execution completes.

After completion, confirm that the following files are created:
- CV_movies.pkl  
- CV_vectors.pkl  
- CV_vectorizer.pkl  
- CV_similarity.pkl  
- TFIDF_movies.pkl  
- TFIDF_vectors.pkl  
- TFIDF_vectorizer.pkl  
- TFIDF_similarity.pkl  

**Step 2:**  
Run `1.2.Content_based_Recommender.py`.

**Step 3:**  
When prompted, enter a movie name exactly as it appears in the dataset  
(example: *Hulk*).

**Step 4:**  
Press **Enter**.

**Step 5:**  
The system will display:
- Recommended movies in the terminal  
- A similarity comparison chart  

---

# Collaborative Filtering Recommendation System

## How to Run

**Step 1:**  
Run `Collaborative_Filtering.py`.

**Step 2:**  
An RMSE comparison chart will be displayed.

**Step 3:**  
Close the RMSE chart window.

**Step 4:**  
After closing the chart, enter a **user ID** when prompted.

**Step 5:**  
Press **Enter**.

**Step 6:**  
The system will display:
- Recommended movies in the terminal  
- Results from User–User and Item–Item filtering  

---

# Popularity-Based Recommendation System

## How to Run

**Step 1:**  
Run `Popularity_Based.py`.

**Step 2:**  
The system will automatically:
- Rank movies by popularity  
- Display the top movies in the terminal  

**Step 3:**  
A popularity chart will be displayed.

**Step 4:**  
Close the chart to finish execution.

---

# Hybrid Recommendation System

## How to Run

**Step 1:**  
Run `Hybrid_Based_Recommender.py`.

**Step 2:**  
The system will automatically:
- Calculate weighted ratings  
- Generate hybrid scores  
- Rank movies  

**Step 3:**  
The system will display:
- Top-ranked movies in the terminal  
- RMSE and comparison charts  

**Step 4:**  
Close the charts to complete execution.

---

# Latent Factor Recommendation System (SVD)

## How to Run

**Step 1:**  
Run `5.1.Latent_Model.py` to train the SVD model.

**Step 2:**  
After training completes, run `5.2.Latent_Recommender.py`.

**Step 3:**  
When prompted, enter a **user ID**.

**Step 4:**  
Press **Enter**.

**Step 5:**  
The system will display:
- Personalized movie recommendations  
- RMSE results  
- An error distribution chart
---
## Contributors

- [Devanshu Prajapati](https://github.com/Devanshu1013)
- [Taran Veer Singh](https://github.com/taran2812)

