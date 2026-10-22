# cake-recommendation

AI Bakery Recommendation & Cake Design Assistant (Femmora Bakes) - built with Streamlit.

## Files
- `app.py` - main Streamlit app
- `cake_render.py` - colour preview drawing (SVG)
- `requirements.txt` - Python packages
- `cake_photos/` - your real cake photos (file names like `birthdaycake1.jpg`, `blackforest1.jpg`)
- `.streamlit/secrets.toml.example` - copy the keys into Streamlit Cloud > App settings > Secrets

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Push all files to GitHub.
2. Streamlit Community Cloud > New app > select this repo > main file `app.py`.
3. Add your secrets (see `.streamlit/secrets.toml.example`).

Never upload your real `secrets.toml` to GitHub.
