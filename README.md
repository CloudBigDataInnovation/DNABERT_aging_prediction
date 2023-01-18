# DNABERT_aging_prediction



## Install

```bash
conda create -n dnabert python=3.8
conda activate dnabert
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge matplotlib
conda install -c anaconda pandas
conda install -c pytorch captum 

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

## Model 

Download the parameters of the model from [download model](https://drive.google.com/file/d/1djqsJY4_bTUe_th8HHpx5KsxSFx8st53/view?usp=share_link).

## Usage

contigs.fasta: sequence with 4096bp

```bash
python DNABERT_prediction.py -i contigs.fasta --model DNABERT_model.tar -o output
```

## Output

- predictions.csv:  predictions of the input sequence

- forward/reverse.csv: attribution score calculated using (https://github.com/pytorch/captum).  (Positive attribution score means that the input in that particular position positively contributed to the final prediction and negative means the opposite.)

- forward/reverse.csv: visualization of the attribution score
