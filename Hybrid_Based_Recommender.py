# Hybrid Recommendation System.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Loading datasets movies metadata, credits, keywords 
# (Note: The data files are under archive folder).
movies = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
credits = pd.read_csv('archive/credits.csv')
keywords = pd.read_csv('archive/keywords.csv')

# Keeping only required columns.
movies = movies[['id','title','original_title', 'overview','vote_average', 'vote_count', 'popularity', 'genres']]
credits = credits[['id','cast', 'crew']]
keywords = keywords[['id', 'keywords']]

# Converting all the id's and required field into numeric.
def convert_to_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

movies = convert_to_numeric(movies, ['id', 'popularity', 'vote_average', 'vote_count'])
credits = convert_to_numeric(credits, ['id'])
keywords = convert_to_numeric(keywords, ['id'])


# Removing missing values and then merging(left join) three datasets on basis on id.
movies = movies.dropna(subset=['id'])
movies_final = movies.merge(credits, on='id', how='left')
movies_final = movies_final.merge(keywords, on='id', how='left')
movies_final = movies_final.dropna(subset=['popularity', 'vote_average', 'vote_count'])

# Claculating the weighted average score using the formula: ((R * v) + (C * m)) / (v + m)
v = movies_final['vote_count']
R = movies_final['vote_average']
C = movies_final['vote_average'].mean()
m = movies_final['vote_count'].quantile(0.70)
movies_final['weighted_average'] = ((R * v) + (C * m)) / (v + m)

# Normalize the weighted_average and popularity so that their values fall between 0 & 1.
scaler = MinMaxScaler()
s_features = scaler.fit_transform(movies_final[['weighted_average', 'popularity']])
movie_norm = pd.DataFrame(s_features, columns=['weighted_average', 'popularity'])

# Compute blended score: weighted_average (80%) + popularity (20%)
# Can be adjusted as per requirement but here I want to give more importance to weighted average as the popularity based rec system is already there.
movie_norm['score'] = 0.8 * movie_norm['weighted_average'] + 0.2 * movie_norm['popularity']
movies_final[['weighted_average', 'popularity', 'score']] = movie_norm
movies_hybrid = movies_final.sort_values('score', ascending=False).head(10)

print("\nTop 10 Movies")
for i, title in enumerate(movies_hybrid['original_title'].tolist(), start=1):
    print(f"{i}. {title} (Score: {movies_hybrid['score'].iloc[i-1]:.4f})")

# Chart of Hybrid Recommendation System which shows Top 10 rec movies.
plt.figure(figsize=(12, 6))
sns.barplot(x='score', y='original_title', data=movies_hybrid, palette='viridis')
plt.title('Top 10 Movies (Popularity + Weighted Average)', fontsize=14)
plt.xlabel('Blended Score', fontsize=12)
plt.ylabel('Movie Title', fontsize=12)
plt.tight_layout()
plt.show()

# RMSE and plot of RMSE
movies_clean = movies_final.dropna(subset=['vote_average', 'weighted_average', 'score'])
rmse_weighted = np.sqrt(mean_squared_error(movies_clean['vote_average'], movies_clean['weighted_average']))
rmse_blend = np.sqrt(mean_squared_error(movies_clean['vote_average'], movies_clean['score']))
print("\nRMSE of Weighted Average:", rmse_weighted)
print("RMSE of Blended Score: ", rmse_blend)

plt.figure(figsize=(8, 5))
methods = ['Weighted Avg', 'Blended Score']
rmse_values = [rmse_weighted, rmse_blend]
rmse_df = pd.DataFrame({'Method': methods, 'RMSE': rmse_values})
sns.barplot(x='Method', y='RMSE', data=rmse_df, hue='Method', dodge=False, palette=["#EC700B", "#0A59CF"])
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE Comparison: Weighted Average vs Blended Score", fontsize=14)
for i, v in enumerate(rmse_values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# This code will first give the list of top 10 Movies based on the blended score.
# 2nd thing it will generate chart of the movies as well.
# 3rd it will print the RMSE score of weighted and Blended score.
# 4th it will show the bar chart of RMSE.