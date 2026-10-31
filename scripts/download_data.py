"""Download the AMP Scanner v2 FASTA files (pinned commit) into data/raw/."""

from ampred.data import download

if __name__ == "__main__":
    print("saved to", download("data/raw"))
