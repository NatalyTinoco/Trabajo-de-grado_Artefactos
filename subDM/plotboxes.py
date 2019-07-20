# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:15:07 2019

"""

def plot_boxes(ax, boxes):
    
    import seaborn as sns 
    
    color_pal = sns.color_palette('hls', n_colors = 7)
    
    for b in boxes:
        cls, x1, y1, x2, y2 = b
        if cls == 0 or cls == 3:
            ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1],lw=2, color=color_pal[int(cls)])
    return []