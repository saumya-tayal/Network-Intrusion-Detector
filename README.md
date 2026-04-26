📌 Overview
This project implements a machine learning-based Network Intrusion Detection System (NIDS) to classify network traffic as **Normal** or **Attack** using the NSL-KDD dataset.

🎯 Features
* Data preprocessing and encoding
* Random Forest classifier
* Threshold tuning using Precision-Recall curve
* Risk-based alerts (Low / Medium / High)
* Interactive UI using Streamlit

⚙️ Tech Stack
* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit

🚀 How to Run
pip install pandas numpy scikit-learn streamlit
python -m streamlit run main.py

📊 Performance
* Accuracy: ~92%
* High recall (~99%) for attack detection
* Reduced false negatives using threshold tuning

⚠️ Note
System prioritizes detecting attacks (high recall), which may increase false alarms.
