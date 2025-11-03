# 🍄 Mushroom Classifier

We used Streamlit + scikit-learn  



- We trained a simple model on the UCI Mushroom dataset.  
- The app is a Streamlit site with dropdowns for mushroom attributes.  
- You click **Predict**, it returns a label + probabilities.  
- Code + steps are in the repo so anyone can re-run it.


## The Idea
Take tabular mushroom data (things like cap color, odor, gill size) and classify mushrooms as **edible** or **poisonous**. Keep it simple, reproducible, and easy to demo.


## Data
- **Source:** UCI ML Repository → *Mushroom* dataset (8,124 rows, 22 categorical features, target = `e`/`p`).  
- **Missing stuff:** `stalk-root` has lots of `?`. We treat that as missing and impute the most frequent value.  
- **Split:** Stratified train/test, 75/25, `random_state=42`.


## Model
A scikit-learn Pipeline:
- `SimpleImputer(most_frequent)` → fill missing
- `OneHotEncoder(handle_unknown="ignore")` → turn categories into numbers
- `LogisticRegression(max_iter=1000)` → fast, works great here

We also considered classic baselines (KNN/RandomForest), but LR is tiny and solid.



## Results
After training on the split above, we got:
- **Accuracy:**  0.999507631708518
- **F1 (poisonous=1):** 0.9994890137966275

## How to Run
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train.py   
python -m streamlit run app.py
```


## The App
- **Predict tab:** Dropdowns with the original UCI letter codes.  
- **Info tab:** A info tab so you can see what each letter means.  


## What’s in the Repo
app.py            # Streamlit UI (Predict + Info)
train.py          # trains model and prints metrics
requirements.txt  # dependencies


## Credits
- UCI ML Repository — Mushroom dataset  
- scikit-learn, Streamlit, Pandas, NumPy
- chat
