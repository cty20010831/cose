# thesis

This directory contains the code for deriving flexibility metrics from the incomplete shape drawing task using the CoSE model. 

## Virtual Environment
Create a virtual environment:
```bash
conda env create -f environment.yml
```

Activate the virtual environment:
```bash
conda activate cose
```

You can also create a virtual environment using `venv` and install the required packages:
```bash
pip install -r requirements.txt
```

## Pipeline
First, load each participant's drawing (for all the three groups of drawings) in ndjson and store them into three separate large ndjson files containing all participants' drawings in the respective group (under `/data` directory).
```bash
python load_drawings.py
```

Next, run data preprocessing on each group of incomplete shape (stimuli) to convert .ndjson data into the format CoSE expects and store them in tfrecords.

```bash
python data_preprocessing.py
```

Next, run the CoSE model on the preprocessed data to derive flexibility metrics.
```bash
python main.py
```