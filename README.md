# Home Advantage in Football — Bias or Real?

A machine learning that investigates whether the well-known *home advantage* in football is a genuine effect of playing at home before a supportive crowd, or whether it is largely explained by differences in team strength. A **Decision Tree Classifier** is trained to predict the outcome of a match, and its feature importances are analysed to determine which factors most influence the result.

## Overview

The project frames home advantage as a supervised classification problem. Each match is described by a set of pre-match conditions, and the model predicts the match result from the home team's perspective:

- `home_win`
- `draw`
- `away_win`

By comparing the model against a simple baseline and inspecting which features it relies on, the project separates the part of home advantage that comes from **team strength** (Elo ratings) from the part that comes from **playing at home** (crowd, travel, venue familiarity).

## Dataset

The data comes from the Kaggle dataset [*Home Advantage in Football — Bias or Real*](https://www.kaggle.com/datasets/sergionefedov/home-advantage-in-football-bias-or-real) and is downloaded automatically in the notebook using `kagglehub`.

- **50,000 matches** drawn from several major leagues (English Premier League, La Liga, Serie A, Bundesliga, and others).
- **25 columns** describing each match.

The columns fall into four groups, which matters because some are only known *after* the match and would cause data leakage if used as inputs:

| Group | Example columns | Used as model input? |
|-------|-----------------|----------------------|
| Team strength | `elo_home`, `elo_away`, `elo_diff` | Yes |
| Crowd conditions | `crowd_present`, `stadium_capacity`, `attendance_pct`, `crowd_size` | Yes |
| Travel / context | `travel_distance_km`, `timezones_crossed`, `is_derby`, `venue_familiarity`, `referee_bias_susceptibility`, `league` | Yes |
| In-match stats | `fouls_home`, `yellow_away`, `red_home`, ... | No (known only during the match) |
| Outcome / leakage | `goals_home`, `goals_away`, `home_goal_diff`, `home_points` | No (dropped) |

**Target variable:** `home_result`

## Project structure

```
.
├── Mini_Project_Classification_Algorithms_Exploration.ipynb   # main notebook
└── README.md
```

## Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- kagglehub

Install everything with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
```

## How to run

1. Clone or download this repository.
2. Open the notebook:

   ```bash
   jupyter notebook Mini_Project_Classification_Algorithms_Exploration.ipynb
   ```

   (It also runs in Google Colab without changes.)
3. Run the cells in order. The dataset downloads automatically through `kagglehub` on the first run and is cached afterwards. A Kaggle account may be required for the download.

## Methodology

1. **Import libraries and load the dataset** via `kagglehub`.
2. **Exploratory Data Analysis (EDA)** — check data types, missing values, and duplicates; plot the target distribution, feature distributions, and the relationship of each feature with the home win rate; inspect a correlation heatmap.
3. **Data preprocessing** — drop the leakage columns and `match_id`, one-hot encode the categorical `league` column, and split the data into 80% training and 20% testing sets (stratified on the target).
4. **Model training** — a `DecisionTreeClassifier` built inside a scikit-learn `Pipeline` with a `ColumnTransformer` for encoding. Key settings: `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`, and `class_weight="balanced"` to handle the class imbalance.
5. **Evaluation** — compare against a `DummyClassifier` baseline (always predicts the most common result) using accuracy, macro F1, a per-class classification report, and a confusion matrix.

### Decision tree background

A Decision Tree chooses each split by making the resulting child nodes as *pure* as possible. The notebook explains the three measures behind this:

- **Gini impurity** — the default criterion in scikit-learn.
- **Entropy** — a measure of disorder in a node.
- **Information gain** — the reduction in impurity achieved by a split; the tree picks the split that maximises it.

Each leaf then predicts the majority class among its samples.

## Key findings

- **Home advantage is real.** Home teams win about **47%** of matches versus about **29%** for away teams. Even when two teams are evenly matched on Elo, the home team still wins far more often.
- **Team strength (`elo_diff`) is the single strongest predictor.** When the home team is much stronger it wins ~74% of the time; when much weaker, only ~21%.
- **The crowd matters, but modestly.** Home teams win ~48% with a crowd versus ~43% without. With a crowd present, away teams are also called for more fouls and yellow cards, suggesting the crowd influences referee decisions.
- **Travel distance helps the home team.** The farther the away team travels, the higher the home win rate (~41% for the shortest trips versus ~55% for the longest).
- **Derbies shrink the advantage.** Local rivals travel little and know the venue, so the home win rate drops to ~43%.
- **Draws are the hardest class to predict**, as they do not depend clearly on any single feature.
- **Baseline:** always predicting `home_win` gives ~47% accuracy. The model is only useful if it beats this.
