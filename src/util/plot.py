import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from torchvision import transforms

from scikitplot.metrics import plot_confusion_matrix, plot_roc

def create_fig_of_confmat(confmat, class_num):
    df_cm = pd.DataFrame(
        confmat, 
        index=np.arange(class_num),
        columns=np.arange(class_num)
    )
    plt.figure()
    sns.set(font_scale=1.2)
    fig = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d').get_figure()
    fig.canvas.draw()
    img = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    c = len(img) // (w * h)
    img = np.frombuffer(img, dtype=np.uint8).reshape(h, w, c)
    plt.close(fig)
    return img

 
def show_img_imnet(dataset, save_path="./data/sample_show.png"):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    plt.figure(figsize=(15, 3))
    for i in range(5):
        image, label = dataset[i]
        image = inv_normalize(image)
        image = image.permute(1, 2, 0)
        
        plt.subplot(1, 5, i+1)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.imshow(image)
    plt.tight_layout()
    plt.savefig(save_path)