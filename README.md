# Text-De-toxification

This project is creating for text-detoxification task. The structure is folowing:
├── README.md # The top-level README
│
├── data 
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints(to big to loag in git)
│
├── notebooks    #  Jupyter notebooks. First for data exploration and second for training and test model        
│ 
│
├── reports      #Two txt files with description of model and specific task
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
