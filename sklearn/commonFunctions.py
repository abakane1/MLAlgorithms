import os
import tarfile
from six.moves import urllib
# To plot pretty figures
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
PROJECT_ROOT_DIR = ""


def saveFig(fig_id, CHAPTER_ID, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def fetchData(url, path, tgzName):
    if not os.path.isdir(path):
        os.makedirs(path)
    tgzPath = os.path.join(path, tgzName)
    urllib.request.urlretrieve(url, tgzPath)
    tgz = tarfile.open(tgzPath)
    tgz.extractall(path=path)
    tgz.close()

def loadData (path,csvName):
    return pd.read_csv(os.path.join(path, csvName))
