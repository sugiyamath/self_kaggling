import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./train.csv", header=None, error_bad_lines=False, encoding="latin1")
    with open("spm_data.txt", "w") as f:
        f.write('\n'.join(df[5].tolist()))
