import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Dataset and keeping the required coulmns
movies = pd.read_csv('archive/movies_metadata.csv', low_memory=False)
movies = movies[['id', 'title', 'original_title', 'popularity']]

# Converting it to numeric columns and dropping missing values
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
movies = movies.dropna(subset=['id', 'popularity'])

# Get top 10 popular movies and printing it.
top_10_popular = movies.sort_values('popularity', ascending=False).head(10)
print("Top 10 Most Popular Movies:")
for i, title in enumerate(top_10_popular['original_title'], start=1):
    print(f"{i}. {title} (Popularity: {top_10_popular['popularity'].iloc[i-1]:.2f})")

# Plot the top 10 popular movies
plt.figure(figsize=(12, 6))
sns.barplot(x='popularity',y='original_title',data=top_10_popular,color="#0A59CF")
plt.title('Top 10 Most Popular Movies')
plt.xlabel('Popularity Score')
plt.ylabel('Movie Title')
plt.tight_layout()
plt.show()
