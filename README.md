# Weather Dashboard

An interactive weather dashboard built with Streamlit and Plotly, featuring a 3D globe.

## How to run

1. **Download** this repository (green **Code** button, then **Download ZIP**) and extract it.
2. **Unzip `data.zip`**. This is required. It contains `weather_data.csv`, which the app needs.
   Extract it so that `weather_data.csv` sits in the same folder as `app.py`.
3. **Open the folder in VS Code** (File > Open Folder).
4. **Open the terminal** in VS Code (Terminal > New Terminal) and install the requirements:

   ```
   pip install -r requirements.txt
   ```

5. **Run the app:**

   ```
   streamlit run app.py
   ```

6. Your browser will open the dashboard automatically. If not, open the link shown in the terminal (usually http://localhost:8501).

## Folder structure after unzipping

```
weather-dashboard/
├── app.py
├── requirements.txt
├── README.md
├── data.zip
└── weather_data.csv   <-- appears after you unzip data.zip
```

## Notes

- If you see a "file not found" error for `weather_data.csv`, you most likely forgot to unzip `data.zip`.
