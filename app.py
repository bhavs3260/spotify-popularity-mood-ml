import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Spotify Hit and Mood Predictor", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")

# ---- EDIT THIS LINE: paste your own "Best settings" from the tuning cell in Kaggle ----
BEST_PARAMS = {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5}

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
MOOD_FEATURES = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness"]

# label: (min, max, default, step)
SLIDERS = {
    "danceability": (0.0, 1.0, 0.5, 0.01),
    "energy": (0.0, 1.0, 0.6, 0.01),
    "loudness": (-40.0, 2.0, -8.0, 0.5),
    "speechiness": (0.0, 1.0, 0.05, 0.01),
    "acousticness": (0.0, 1.0, 0.3, 0.01),
    "instrumentalness": (0.0, 1.0, 0.0, 0.01),
    "liveness": (0.0, 1.0, 0.15, 0.01),
    "valence": (0.0, 1.0, 0.5, 0.01),
    "tempo": (40.0, 220.0, 120.0, 1.0),
    "duration_sec": (30.0, 600.0, 210.0, 5.0),
}


@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna()
    df = df.drop_duplicates(subset="track_id").reset_index(drop=True)
    df["is_hit"] = (df["popularity"] >= 50).astype(int)
    return df


def name_moods(profile):
    z = (profile - profile.mean()) / profile.std()
    left = list(profile.index)
    names = {}
    c = (z.loc[left, "acousticness"] - z.loc[left, "energy"]).idxmax()
    names[int(c)] = "Calm Acoustic"
    left.remove(c)
    c = (z.loc[left, "energy"] + z.loc[left, "loudness"] + z.loc[left, "danceability"]).idxmax()
    names[int(c)] = "Upbeat Dance"
    left.remove(c)
    c = z.loc[left, "valence"].idxmin()
    names[int(c)] = "Dark and Intense"
    left.remove(c)
    for c in left:
        names[int(c)] = "Mellow Groove"
    return names


@st.cache_resource(show_spinner="Training the models (about 30 seconds the first time)...")
def train(_df):
    df = _df.copy()
    X, y = df[FEATURES], df["is_hit"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(**BEST_PARAMS, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    prob = rf.predict_proba(X_te_s)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, prob),
    }
    importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()

    mood_scaler = StandardScaler()
    Xm = mood_scaler.fit_transform(df[MOOD_FEATURES])
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xm)
    df["mood_cluster"] = km.labels_
    profile = df.groupby("mood_cluster")[MOOD_FEATURES].mean()
    names = name_moods(profile)
    df["mood"] = df["mood_cluster"].map(names)

    pca = PCA(n_components=2, random_state=42).fit(Xm)
    idx = np.random.RandomState(1).choice(len(df), min(6000, len(df)), replace=False)
    coords = pca.transform(Xm[idx])
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "mood": df["mood"].iloc[idx].values})

    return {
        "rf": rf, "scaler": scaler, "metrics": metrics, "importance": importance,
        "mood_scaler": mood_scaler, "km": km, "pca": pca, "names": names,
        "plot_df": plot_df, "hit_rate": df.groupby("mood")["is_hit"].mean().sort_values(ascending=False),
        "mood_counts": df["mood"].value_counts(), "n_songs": len(df), "hit_share": df["is_hit"].mean(),
    }


try:
    data = load_data()
except FileNotFoundError:
    st.error("dataset.csv was not found. Put dataset.csv in the same folder as app.py.")
    st.stop()

M = train(data)

for key, (lo, hi, default, step) in SLIDERS.items():
    st.session_state.setdefault(key, default)


def load_song(idx):
    row = data.loc[idx]
    for f in FEATURES:
        if f == "duration_ms":
            lo, hi = SLIDERS["duration_sec"][0], SLIDERS["duration_sec"][1]
            st.session_state["duration_sec"] = float(np.clip(row["duration_ms"] / 1000, lo, hi))
        else:
            lo, hi = SLIDERS[f][0], SLIDERS[f][1]
            st.session_state[f] = float(np.clip(row[f], lo, hi))
    st.session_state["picked"] = (str(row.get("track_name", "Unknown")), str(row.get("artists", "Unknown")), int(row["popularity"]))


def load_random_song():
    load_song(int(data.sample(1).index[0]))


def reset_song():
    for key, spec in SLIDERS.items():
        st.session_state[key] = spec[2]
    st.session_state.pop("picked", None)


st.title("🎵 Decoding Music: Hit Predictor and Mood Finder")
st.caption("Machine learning project on the Spotify Tracks dataset. Random Forest predicts whether a song is a hit, and K-Means finds its mood.")

st.subheader("1. Pick a real song, or move the sliders yourself")
col_a, col_b, col_c = st.columns([2, 3, 2])
with col_a:
    query = st.text_input("🔍 Search a song or artist")
with col_b:
    if query:
        mask = (data["track_name"].str.contains(query, case=False, na=False, regex=False)
                | data["artists"].str.contains(query, case=False, na=False, regex=False))
        matches = data[mask].sort_values("popularity", ascending=False).head(30)
        if len(matches) == 0:
            st.caption("No match found. Try another spelling.")
        else:
            labels = {i: f"{r['track_name']} - {r['artists']} (popularity {int(r['popularity'])})" for i, r in matches.iterrows()}
            choice = st.selectbox("Pick a match", list(matches.index), format_func=lambda i: labels[i])
            st.button("Load this song", on_click=load_song, args=(choice,))
    else:
        st.caption("Type a song or artist name, then pick a match and click Load this song.")
with col_c:
    st.button("🎲 Load a random real song", on_click=load_random_song, width="stretch")
    st.button("Reset sliders", on_click=reset_song, width="stretch")

with st.expander("🎚️ Audio features (move these to see how the prediction changes)", expanded=True):
    slider_cols = st.columns(5)
    for i, (key, (lo, hi, default, step)) in enumerate(SLIDERS.items()):
        label = "duration (seconds)" if key == "duration_sec" else key
        with slider_cols[i % 5]:
            st.slider(label, lo, hi, step=step, key=key)

vals = {f: st.session_state[f] for f in FEATURES if f != "duration_ms"}
vals["duration_ms"] = st.session_state["duration_sec"] * 1000
x = pd.DataFrame([vals])[FEATURES]

prob = float(M["rf"].predict_proba(M["scaler"].transform(x))[0, 1])
xm = M["mood_scaler"].transform(x[MOOD_FEATURES])
cluster = int(M["km"].predict(xm)[0])
mood = M["names"][cluster]
pt = M["pca"].transform(xm)[0]

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Mood map", "Model report", "About and limits"])

with tab1:
    if "picked" in st.session_state:
        title, artist, pop = st.session_state["picked"]
        st.info(f"Loaded song: **{title}** by {artist}. Its real popularity score is **{pop}** (hit if 50 or more).")
    c1, c2 = st.columns(2)
    c1.metric("Chance of being a hit", f"{prob * 100:.0f}%")
    c1.progress(min(max(prob, 0.0), 1.0))
    if prob >= 0.5:
        c1.success("Model says: likely a hit")
    else:
        c1.warning("Model says: probably not a hit")
    c2.metric("Mood", mood)
    c2.write("Mood comes from K-Means clustering on danceability, energy, valence, acousticness, tempo and loudness.")
    st.caption(f"For reference, about {M['hit_share'] * 100:.0f}% of all songs in the data are hits. Move the sliders and watch the chance change.")

with tab2:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in M["plot_df"].groupby("mood"):
        ax.scatter(g["x"], g["y"], s=8, alpha=0.4, label=name)
    ax.scatter([pt[0]], [pt[1]], s=250, marker="*", color="black", label="Your song", zorder=5)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Song mood map (PCA)")
    ax.legend(markerscale=2)
    st.pyplot(fig)
    st.write("PCA squashes the 6 mood features into 2 so the songs can be drawn on a flat map. The black star is the song described by the sliders.")

with tab3:
    st.subheader("Hit prediction: Random Forest on the held-out test set")
    st.dataframe(pd.DataFrame(M["metrics"], index=["Score"]).round(3), width="stretch")
    left, right = st.columns(2)
    with left:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        M["importance"].plot(kind="barh", ax=ax2, color="teal")
        ax2.set_title("Feature importance")
        plt.tight_layout()
        st.pyplot(fig2)
    with right:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        M["hit_rate"].plot(kind="bar", ax=ax3, color="seagreen")
        ax3.set_title("Share of hits in each mood")
        ax3.set_ylabel("Hit rate")
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig3)
    st.write("Songs per mood:", M["mood_counts"].to_dict())

with tab4:
    st.markdown(
        f"""
**Data:** {M['n_songs']:,} unique songs after removing missing values and duplicate track IDs. A song is a hit if its popularity score is 50 or more.

**Honest limits**
- Only audio features are used. Popularity also depends on the artist's fame, marketing and playlists, which are not in this data, so the prediction is only a weak signal.
- The classes are imbalanced, so accuracy alone is misleading. Look at ROC-AUC, recall and F1 in the Model report tab.
- Moods are discovered by clustering, so the names are my interpretation of each cluster's average values.
"""
    )
