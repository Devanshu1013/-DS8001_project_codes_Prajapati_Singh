# Imported necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# Loaded the ratings_small dataset that contains userId, movieId, and rating.
ratings = pd.read_csv("archive/ratings_small.csv")

# Split the dataset into training and testing sets to evaluate predictions on unseen data.
X_train, X_test = train_test_split(ratings, test_size=0.30, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Created the user–item matrix where rows were users and columns were movies with missing values filled with 0.
user_item_train = X_train.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Calculated cosine similarity between users for user–user collaborative filtering.
user_similarity = cosine_similarity(user_item_train)
user_similarity[np.isnan(user_similarity)] = 0

# Generated user-based rating predictions using similarity-weighted sums.
user_pred = np.dot(user_similarity, user_item_train)

# Normalized the predictions by dividing by the total similarity for each user.
sim_sums = np.array([np.abs(user_similarity).sum(axis=1)]).T
user_pred_normalized = user_pred / (sim_sums + 1e-8)

# Converted normalized predictions into a DataFrame for easy lookup.
user_pred_df = pd.DataFrame(
    user_pred_normalized,
    index=user_item_train.index,
    columns=user_item_train.columns
)

# Constructed the item–user matrix by transposing the user–item matrix.
item_user_train = user_item_train.T

# Calculated cosine similarity between items for item–item collaborative filtering.
item_similarity = cosine_similarity(item_user_train)
item_similarity[np.isnan(item_similarity)] = 0

# Generated item-based rating predictions by multiplying ratings with item similarity.
item_pred = np.dot(user_item_train, item_similarity)

# Normalized item-based predictions using similarity sums.
sim_sums_it = np.array([np.abs(item_similarity).sum(axis=1)])
item_pred_normalized = item_pred / (sim_sums_it + 1e-8)

# Converted item-based predictions into a DataFrame for easy lookup.
item_pred_df = pd.DataFrame(
    item_pred_normalized,
    index=user_item_train.index,
    columns=user_item_train.columns
)

# Defined a function that extracted true and predicted ratings for RMSE evaluation.
def get_predicted(test_df, pred_matrix):
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u = row["userId"]
        m = row["movieId"]
        actual = row["rating"]
        if u in pred_matrix.index and m in pred_matrix.columns:
            pred = pred_matrix.loc[u, m]
            y_true.append(actual)
            y_pred.append(pred)
    return y_true, y_pred

# Retrieved true and predicted values for user-based and item-based CF.
y_true_u, y_pred_u = get_predicted(X_test, user_pred_df)
y_true_i, y_pred_i = get_predicted(X_test, item_pred_df)

# Calculated RMSE for both CF methods.
rmse_user = np.sqrt(mean_squared_error(y_true_u, y_pred_u))
rmse_item = np.sqrt(mean_squared_error(y_true_i, y_pred_i))

print("\nRMSE Results:")
print("User–User CF RMSE:", rmse_user)
print("Item–Item CF RMSE:", rmse_item)

# Plotted RMSE comparison between user–user and item–item CF.
plt.figure(figsize=(6,4))
methods = ["User–User CF", "Item–Item CF"]
rmses = [rmse_user, rmse_item]

plt.bar(methods, rmses)
plt.ylabel("RMSE")
plt.title("RMSE: User–User vs Item–Item Collaborative Filtering")
plt.ylim(0, max(rmses) + 1)

# Displayed RMSE values above the bars.
for i, v in enumerate(rmses):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.show()

# Loaded movie metadata files to map movieId to titles.
movies = pd.read_csv("archive/movies_metadata.csv", low_memory=False)
links_small = pd.read_csv("archive/links_small.csv")

# Converted ID columns into numeric so merging worked properly.
movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
links_small["movieId"] = pd.to_numeric(links_small["movieId"], errors="coerce")
links_small["tmdbId"] = pd.to_numeric(links_small["tmdbId"], errors="coerce")

# Removed rows with missing IDs.
movies = movies.dropna(subset=["id"])
links_small = links_small.dropna(subset=["movieId", "tmdbId"])

# Defined a function to return top-N user–user CF recommendations.
def get_top_n_user_based(user_final_ratings, user_id, n=5):
    if user_id not in user_final_ratings.index:
        print(f"User {user_id} not found in training data.")
        return None

    # Selected the movies with the highest predicted ratings for this user.
    top_n = user_final_ratings.loc[user_id].sort_values(ascending=False)[:n]
    top_n = pd.DataFrame(top_n).reset_index()
    top_n.columns = ["movieId", "predicted_rating"]

    # Merged movieId with TMDB metadata to retrieve movie titles.
    top_n = top_n.merge(links_small, on="movieId", how="left")
    top_n = top_n.merge(
        movies[["id", "original_title"]],
        left_on="tmdbId",
        right_on="id",
        how="left"
    )

    return top_n[["original_title", "predicted_rating"]]

# Defined a function to return item–item CF recommendations while excluding already-rated movies.
def get_top_n_item_based(user_id, pred_matrix, train_df, movies, links_small, n=5):
    if user_id not in pred_matrix.index:
        print(f"User {user_id} not found in training data.")
        return None

    # Sorted predictions in descending order for this user.
    preds = pred_matrix.loc[user_id].sort_values(ascending=False)

    # Removed movies the user already rated.
    seen_movies = train_df[train_df["userId"] == user_id]["movieId"].tolist()
    preds = preds[~preds.index.isin(seen_movies)]

    # Selected top-N unseen movies.
    top_n = preds[:n]
    top_n = pd.DataFrame(top_n).reset_index()
    top_n.columns = ["movieId", "predicted_rating"]

    # Merged with metadata to obtain movie titles.
    top_n = top_n.merge(links_small, on="movieId", how="left")
    top_n = top_n.merge(
        movies[["id", "original_title"]],
        left_on="tmdbId",
        right_on="id",
        how="left"
    )

    return top_n[["original_title", "predicted_rating"]]

# Asked the user for an ID and displayed user–user recommendations.
try:
    user_u = int(input("\nEnter user ID for User–User CF recommendation: "))
    print(f"\nTop User–User Recommendations for User {user_u}:")
    print(get_top_n_user_based(user_pred_df, user_u, n=5))
except Exception as e:
    print("Error in User–User CF:", e)

# Asked the user again for item–item recommendations.
try:
    user_i = int(input("\nEnter user ID for Item–Item CF recommendation: "))
    print(f"\nTop Item–Item Recommendations for User {user_i}:")
    print(get_top_n_item_based(user_i, item_pred_df, X_train, movies, links_small, n=5))
except Exception as e:
    print("Error in Item–Item CF:", e)
