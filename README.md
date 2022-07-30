# REDGL
Implementations of model **REDGL**  described in our paper.

## Environment Requirement

The project is implemented using python 3.6 and tested in Linux environment. We use ``anaconda`` to manage our experiment environment.

Our system environment and CUDA version as follows:

```bash
Nvidia A100 CUDA Version: 11.4
```

Our python version and requirements as follows:

- Python 3.6.13
- PyTorch 1.9.0

## Usage

1. Install all the requirements.

2. If there is no folder `data`, you can download it from https://drive.google.com/file/d/1SfNg7zCJKNb3ArU6ACJtgM7NAjNhQyu_/view?usp=sharing.

3. Train the model using the Python script `main.py` .

   You can run the following command to train the model **REDGL** on the Tmall dataset:

   ```bash
   python main.py --dataset Clothing
   ```
