# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:02:36 2024

@author: webbe

figures for PCSWMM 1D paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

path_to_save = r"C:/Users/webbe/Box/Marissa's Research/MARISA 2.0/Figuresfor1DPaper/"
path = r"C:/Users/webbe/Box/Marissa's Research/MARISA 2.0/Figuresfor1DPaper/"

#%% methods

#method to identify the max value of a list of lists and return the index of the max
def identifymaxofmax(frames):
    values = []
    for i in frames:
        values.append(max(i))
    maxindex = values.index(max(values))
    return maxindex

#pull the data for scatter plots from userdefined file and sheet
#used for 10 calibration storms
def generatedataforscatter(path, file, sheetname):
    df = pd.read_excel(path + file, sheetname)
    #drop first three rows
    df = df.drop([0, 1, 2], axis = 0)
    df.columns = df.iloc[0]
    df = df.drop([3], axis = 0).reset_index()
    x = list(df['Obs'].iloc[:-5])
    prior = list(df['Basecase'].iloc[:-5])
    y_self = list(df['CaltoSelf'].iloc[:-5])
    y_up = list(df['CaltoUpstream'].iloc[:-5])
    y_down = list(df['CaltoDownstream'].iloc[:-5])
    return [x, prior, y_self, y_up, y_down]

#pull the data for scatter plots from userdefined file and sheet
#used for 44 validation storms
def generatedataforscatter_44(path, file, sheetname):
    df = pd.read_excel(path + file, sheetname)
    #drop first three rows
    df = df.drop([0, 1, 2], axis = 0)
    df.columns = df.iloc[0]
    df = df.drop([3], axis = 0).reset_index()
    x1 = list(df['Obs'].iloc[:37])
    x2 = list(df['Obs'].iloc[47:54])
    prior1 = list(df['Basecase'].iloc[:37])
    prior2 = list(df['Basecase'].iloc[47:54])
    y_self1 = list(df['CaltoSelf'].iloc[:37])
    y_self2 = list(df['CaltoSelf'].iloc[47:54])
    y_up1 = list(df['CaltoUpstream'].iloc[:37])
    y_up2 = list(df['CaltoUpstream'].iloc[47:54])
    y_down1 = list(df['CaltoDownstream'].iloc[:37])
    y_down2 = list(df['CaltoDownstream'].iloc[47:54])
    return [x1+x2, prior1+prior2, y_self1+y_self2, y_up1+y_up2, y_down1+y_down2]

#calculate RMSE of a based on a list of predictions and targets
def rmse(predictions, targets):
    diff = []
    for i in range(len(predictions)):
        diff.append((predictions[i] - targets[i])**2)
    return np.sqrt(stats.mean(diff))

#method to make multipanel plots
#Top panel = normalized RMSE for each monitoring location, sorted from most upstream to most downstream
#Bottom panel = utility as calculated based on normalized RMSE for each scenario (on the x-axis). 
def multipanelplot6_scatterandsort(df, st, en, ymax1, ymin2, Llist, Ulist, title):
    fig, axs = plt.subplots(2, 6, figsize = (12, 8))
    #Bottom panels = no lines between?
    #df8 = df.iloc[6:12]
    axs[0, 0].scatter(list(range(0, 6)), list(df.iloc[st+0, 8:14]), color = "k")
    axs[0, 0].tick_params(axis='x', labelrotation=90)
    axs[0, 0].set_ylim((0, ymax1))
    axs[0, 0].set_ylabel("Normalized RMSE (NRMSE)")
    axs[0, 0].set_xticks(ticks=list(range(0, 6)))
    axs[0, 0].set_xticklabels(labels=labellist)
    axs[0, 0].set_title(str(Llist[0]))
    #
    axs[0, 1].scatter(list(range(0, 6)), list(df.iloc[st+1, 8:14]), color = "k")
    axs[0, 1].tick_params(axis='x', labelrotation=90)
    axs[0, 1].set_ylim((0, ymax1))
    axs[0, 1].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 1].set_xticks(ticks=list(range(0, 6)))
    axs[0, 1].set_xticklabels(labels=labellist)
    axs[0, 1].set_title(str(Llist[1]))
    #
    axs[0, 2].scatter(list(range(0, 6)), list(df.iloc[st+2, 8:14]), color = "k")
    axs[0, 2].tick_params(axis='x', labelrotation=90)
    axs[0, 2].set_ylim((0, ymax1))
    axs[0, 2].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 2].set_xticks(ticks=list(range(0, 6)))
    axs[0, 2].set_xticklabels(labels=labellist)
    axs[0, 2].set_title(str(Llist[2]))
    #
    axs[0, 5].scatter(list(range(0, 6)), list(df.iloc[st+3, 8:14]), color = "k")
    axs[0, 5].tick_params(axis='x', labelrotation=90)
    axs[0, 5].set_ylim((0, ymax1))
    axs[0, 5].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 5].set_xticks(ticks=list(range(0, 6)))
    axs[0, 5].set_xticklabels(labels=labellist)
    axs[0, 5].set_title(str(Llist[3]))
    #
    axs[0, 3].scatter(list(range(0, 6)), list(df.iloc[st+4, 8:14]), color = "k")
    axs[0, 3].tick_params(axis='x', labelrotation=90)
    axs[0, 3].set_ylim((0, ymax1))
    axs[0, 3].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 3].set_xticks(ticks=list(range(0, 6)))
    axs[0, 3].set_xticklabels(labels=labellist)
    axs[0, 3].set_title(str(Llist[4]))
    #
    axs[0, 4].scatter(list(range(0, 6)), list(df.iloc[st+5, 8:14]), color = "k")
    axs[0, 4].tick_params(axis='x', labelrotation=90)
    axs[0, 4].set_ylim((0, ymax1))
    axs[0, 4].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 4].set_xticks(ticks=list(range(0, 6)))
    axs[0, 4].set_xticklabels(labels=labellist)
    axs[0, 4].set_title(str(Llist[5]))
    #
    axs[1, 0].bar(range(6), df.iloc[st, 14:20], color = bar_colors)
    axs[1, 0].set_ylim((ymin2, 1.5))
    axs[1, 0].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 0].set_ylabel("scaled NRMSE")
    axs[1, 0].set_xlabel(str(Ulist[0]), fontsize=16)
    axs[1, 0].spines['right'].set_visible(False)
    axs[1, 1].bar(range(6), df.iloc[st+1, 14:20], color = bar_colors)
    axs[1, 1].set_ylim((ymin2, 1.5))
    axs[1, 1].spines['left'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)
    axs[1, 1].set_xlabel(str(Ulist[1]), fontsize=16)
    axs[1, 1].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 1].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[1, 2].bar(range(6), df.iloc[st+2, 14:20], color = bar_colors)
    axs[1, 2].set_ylim((ymin2, 1.5))
    axs[1, 2].spines['left'].set_visible(False)
    axs[1, 2].spines['right'].set_visible(False)
    axs[1, 2].set_xlabel(str(Ulist[2]), fontsize=16)
    axs[1, 2].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 2].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[1, 5].bar(range(6), df.iloc[st+3, 14:20], color = bar_colors)
    axs[1, 5].set_ylim((ymin2, 1.5))
    axs[1, 5].spines['left'].set_visible(False)
    axs[1, 4].spines['right'].set_visible(False)
    axs[1, 5].set_xlabel(str(Ulist[3]), fontsize=16)
    axs[1, 5].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 5].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[1, 3].bar(range(6), df.iloc[st+4, 14:20], color = bar_colors)
    axs[1, 3].set_ylim((ymin2, 1.5))
    axs[1, 3].spines['left'].set_visible(False)
    axs[1, 3].spines['right'].set_visible(False)
    axs[1, 3].set_xlabel(str(Ulist[4]), fontsize=16)
    axs[1, 3].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 3].tick_params(axis='y', which='both', left=False, right = False, labelleft =False, labelright = False)
    axs[1, 4].bar(range(6), df.iloc[st+5, 14:20], color = bar_colors)
    axs[1, 4].set_ylim((ymin2, 1.5))
    axs[1, 4].spines['left'].set_visible(False)
    axs[1, 4].set_xlabel(str(Ulist[5]), fontsize=16)
    axs[1, 4].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 4].tick_params(axis='y', which='both', left=False, labelleft =False, right =False, labelright = False)
    plt.savefig(path_to_save + title)
    
#method to make multipanel plots
#Top panel = normalized RMSE for each scenario
#Bottom panel = utility as calculated based on normalized RMSE for each scenario (on the x-axis). 
def multipanelplot5_scatter(df, st, en, ymax1, ymin2, Llist, Ulist, title):
    fig, axs = plt.subplots(2, 4, figsize = (12, 8))#, sharey=True) #can I write this as a method? since I'm doing it 3 times?
    #Bottom panels = no lines between?
    #df_sub = df.iloc[st:en]
    axs[0, 0].scatter(list(range(0, 6)), df.iloc[st+0, 8:14], color = "k")
    axs[0, 0].tick_params(axis='x', labelrotation=90)
    axs[0, 0].set_ylim((0, ymax1))
    axs[0, 0].set_ylabel("Normalized RMSE (NRMSE)")
    axs[0, 0].set_title(str(Llist[0]))
    axs[0, 0].set_xticks(ticks=list(range(0, 6)))
    axs[0, 0].set_xticklabels(labels=labellist)
    axs[0, 1].scatter(list(range(0, 6)), df.iloc[st+1, 8:14], color = "k")
    axs[0, 1].tick_params(axis='x', labelrotation=90)
    axs[0, 1].set_ylim((0, ymax1))
    axs[0, 1].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 1].set_title(str(Llist[1]))
    axs[0, 1].set_xticks(ticks=list(range(0, 6)))
    axs[0, 1].set_xticklabels(labels=labellist)
    axs[0, 2].scatter(list(range(0, 6)), df.iloc[st+3, 8:14], color = "k")
    axs[0, 2].tick_params(axis='x', labelrotation=90)
    axs[0, 2].set_ylim((0, ymax1))
    axs[0, 2].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 2].set_title(str(Llist[2]))
    axs[0, 2].set_xticks(ticks=list(range(0, 6)))
    axs[0, 2].set_xticklabels(labels=labellist)
    axs[0, 3].scatter(list(range(0, 6)), df.iloc[st+4, 8:14], color = "k")
    axs[0, 3].tick_params(axis='x', labelrotation=90)
    axs[0, 3].set_ylim((0, ymax1))
    axs[0, 3].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[0, 3].set_title(str(Llist[3]))
    axs[0, 3].set_xticks(ticks=list(range(0, 6)))
    axs[0, 3].set_xticklabels(labels=labellist)
    axs[1, 0].bar(range(6), df.iloc[st+0, 14:20], color = bar_colors)
    axs[1, 0].set_ylim((ymin2, 1.5))
    axs[1, 0].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 0].set_ylabel("Scaled NRMSE")
    axs[1, 0].spines['right'].set_visible(False)
    axs[1, 0].set_xlabel(str(Ulist[0]), fontsize = 16)
    axs[1, 1].bar(range(6), df.iloc[st+1, 14:20], color = bar_colors)
    axs[1, 1].set_ylim((ymin2, 1.5))
    axs[1, 1].spines['left'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)
    axs[1, 1].set_xlabel(str(Ulist[1]), fontsize = 16)
    axs[1, 1].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 1].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[1, 2].bar(range(6), df.iloc[st+3, 14:20], color = bar_colors)
    axs[1, 2].set_ylim((ymin2, 1.5))
    axs[1, 2].spines['left'].set_visible(False)
    axs[1, 2].spines['right'].set_visible(False)
    axs[1, 2].set_xlabel(str(Ulist[3]), fontsize = 16)
    axs[1, 2].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 2].tick_params(axis='y', which='both', left=False, labelleft =False)
    axs[1, 3].bar(range(6), df.iloc[st+4, 14:20], color = bar_colors)
    axs[1, 3].set_ylim((ymin2, 1.5))
    axs[1, 3].spines['left'].set_visible(False)
    axs[1, 3].set_xlabel(str(Ulist[4]), fontsize = 16)
    axs[1, 3].tick_params(axis='x', which='both', bottom=False, labelbottom =False)
    axs[1, 3].tick_params(axis='y', which='both', left=False, right = False, labelleft =False)
    plt.savefig(path_to_save + title)
    
    
#%% figure S3
#depth (in) and intensity (in/hr) of storms observed in 2022 monitoring period

file_rain = r"PCSWMM_stormsfromraingauges.xlsx"
sheets = ['153135', '153136', 
          '154134', '154135', '154136', '154137', '154138',
          '155135', '155136', '155137', '155138', 
          '156135', '156136', '156137', '156138',
          '157136', '157137', '158137']

colors = plt.cm.get_cmap('tab20b')

pixelrain = []
pixeldur = []
for i in range(len(sheets)):
    df = pd.read_excel(path + file_rain, sheet_name= sheets[i])
    df = df.drop([0], axis = 0)
    df.columns = df.iloc[0]
    df = df.drop([1], axis = 0)
    pixelrain.append(list(df['Total Rainfall (in)'].iloc[:-4]))
    pixeldur.append(list(df['Duration (h)'].iloc[:-4]))

plt.figure()
plt.boxplot(pixelrain)
plt.xlabel("Pixel")
plt.ylabel("Precipitation (in)")
plt.xticks(range(1, 19), sheets, rotation = 90)
plt.tight_layout()
plt.savefig(path_to_save + "boxplot_pptdepth.png")


pixelint = []
for i in range(len(sheets)):
    df = pd.read_excel(path + file_rain, sheet_name= sheets[i])
    df = df.drop([0], axis = 0)
    df.columns = df.iloc[0]
    df = df.drop([1], axis = 0)
    pixelint.append(list(df['Maximum Rainfall (in/hr)'].iloc[:-4]))

plt.figure()
plt.boxplot(pixelint)
plt.xlabel("Pixel")
plt.ylabel("Max Precipitation Intensity (in/hr)")
plt.xticks(range(1, 19), sheets, rotation = 90)
plt.tight_layout()
plt.savefig(path_to_save + "boxplot_pptint.png")


#nested loops to find the average precip and duration for each storm (not for each pixel)
pixelrain_bycell = []
pixeldur_bycell = []
for j in range(len(pixelrain[0])):
    p1 = []
    d1 = []
    for i in range(len(pixelrain)):
        p1.append(pixelrain[i][j])
        d1.append(pixeldur[i][j])
    pixelrain_bycell.append(p1)
    pixeldur_bycell.append(d1)
averagepixelrain = []
averagepixeldur = []
for i in range(len(pixelrain_bycell)):
    averagepixelrain.append(stats.mean(pixelrain_bycell[i]))
    averagepixeldur.append(stats.mean(pixeldur_bycell[i]))

#%% Figure S4

#Scatter plot comparing modeled peak flow to observed peak flow for 44 storms data set
#calibrated to most upstream location (Kedron St) for a single storm (Aug 5)
path_exp1 = path
file = r"PCSWMM1D_parametersExp1.xlsx"
sheet_exp1 = "Mar-Oct_withlimits"
kedron_exp1 = pd.read_excel(path_exp1 + file, sheet_name = sheet_exp1)
kedron_exp1_obs = list(kedron_exp1[kedron_exp1.columns[5]].iloc[1:38]) + list(kedron_exp1[kedron_exp1.columns[5]].iloc[48:55])
kedron_exp1_prior = list(kedron_exp1[kedron_exp1.columns[4]].iloc[1:38]) + list(kedron_exp1[kedron_exp1.columns[4]].iloc[48:55])
kedron_exp1_cal1 = list(kedron_exp1[kedron_exp1.columns[54]].iloc[1:38]) + list(kedron_exp1[kedron_exp1.columns[54]].iloc[48:55])

lists = [kedron_exp1_obs, kedron_exp1_prior, kedron_exp1_cal1]
maxindex = [max(i) for i in lists].index(max([max(i) for i in lists]))

plt.figure()
plt.plot(lists[maxindex], lists[maxindex], c = "grey", linewidth=0.5) #add 1:1 line behind
plt.scatter(kedron_exp1_obs, kedron_exp1_prior, marker = '^', c = "gold", label = "Prior")
plt.scatter(kedron_exp1_obs, kedron_exp1_cal1, marker = 'o', c = "dodgerblue", label = "Cal1")
plt.xlabel("Obs peak flow (mgd)") 
plt.ylabel("Mod peak flow (mgd)") 
plt.axis('square')
plt.legend(loc='lower right')
plt.savefig(path_to_save + "2022storms_Cal10.png")


#%% figure 3
#Scatter plots comparing modeled peak flow to observed peak flow for each monitoring location
#for the prior model (baseline), where the model is calibrated to the monitoring location (x)
#the most upstream location (blue squares) and the most downstream location (purple diamonds). 
#6 panel figure

file2 = r"PCSWMM_multipanelscatterplot.xlsx"
sheetnames = ['A003', 'B023', 'K027', 'P002', 'A009', 'D011']

#all 54 storms
data_k_54 = generatedataforscatter(path, file2, sheetnames[0])
data_h_54 = generatedataforscatter(path, file2, sheetnames[1])
data_sb_54 = generatedataforscatter(path, file2, sheetnames[2])
data_sk_54 = generatedataforscatter(path, file2, sheetnames[3])
data_sl_54 = generatedataforscatter(path, file2, sheetnames[4])
data_df_54 = generatedataforscatter(path, file2, sheetnames[5])

#44 validation storms
data_k = generatedataforscatter_44(path, file2, sheetnames[0])
data_h = generatedataforscatter_44(path, file2, sheetnames[1])
data_sb = generatedataforscatter_44(path, file2, sheetnames[2])
data_sk = generatedataforscatter_44(path, file2, sheetnames[3])
data_sl = generatedataforscatter_44(path, file2, sheetnames[4])
data_df = generatedataforscatter_44(path, file2, sheetnames[5])

RMSE_A003 = []
RMSE_B023 = []
RMSE_K027 = []
RMSE_P002 = []
RMSE_D011 = []
RMSE_A009 = []
for i in range(4):
    RMSE_A003.append(rmse(data_k[0], data_k[i+1]))
    RMSE_B023.append(rmse(data_h[0], data_h[i+1]))
    RMSE_K027.append(rmse(data_sb[0], data_sb[i+1]))
    RMSE_P002.append(rmse(data_sk[0], data_sk[i+1]))
    RMSE_D011.append(rmse(data_sl[0], data_sl[i+1]))
    RMSE_A009.append(rmse(data_df[0], data_df[i+1]))

fig, axs = plt.subplots(2, 3, figsize = (10, 6))
#kedron
axs[0, 0].scatter(data_k[0], data_k[1], marker = "^", c = 'gold')
axs[0, 0].scatter(data_k[0], data_k[2], marker = "x", c = 'k')
axs[0, 0].scatter(data_k[0], data_k[3], marker = "s", c = 'blue', alpha=.5)
axs[0, 0].scatter(data_k[0], data_k[4], marker = "D", c = 'purple', alpha=.5)
axs[0, 0].axis('square')
#axs[0, 0].set_yticks([0, 0.5, 1, 1.5])
axs[0, 0].set_title('a) Kedron')
axs[0, 0].set_ylabel("Mod peak flow (mgd)") 
frames_k = [data_k[0], data_k[1], data_k[2], data_k[3]]
maxindex_k = identifymaxofmax(frames_k)
axs[0, 0].plot(frames_k[maxindex_k], frames_k[maxindex_k], c = "grey", linewidth=0.5) #add 1:1 line behind
#homewood
axs[0, 1].set_title('Homewood')
axs[0, 1].scatter(data_h[0], data_h[1], marker = "^", c = 'gold')
axs[0, 1].scatter(data_h[0], data_h[2], marker = "x", c = 'k')
axs[0, 1].scatter(data_h[0], data_h[3], marker = "s", c = 'blue', alpha=.5)
axs[0, 1].scatter(data_h[0], data_h[4], marker = "D", c = 'purple', alpha=.5)
axs[0, 1].axis('square')
axs[0, 1].set_xticks([0, 2, 4, 6, 8, 10])
axs[0, 1].set_title('b) Homewood')
frames_h = [data_h[0], data_h[1], data_h[2], data_h[3]]
maxindex_h = identifymaxofmax(frames_h)
axs[0, 1].plot(frames_h[maxindex_h], frames_h[maxindex_h], c = "grey", linewidth=0.5) #add 1:1 line behind
#sterrett and bennett
axs[0, 2].scatter(data_sb[0], data_sb[1], marker = "^", c = 'gold')
axs[0, 2].scatter(data_sb[0], data_sb[2], marker = "x", c = 'k')
axs[0, 2].scatter(data_sb[0], data_sb[3], marker = "s", c = 'blue', alpha=.5)
axs[0, 2].scatter(data_sb[0], data_sb[4], marker = "D", c = 'purple', alpha=.5)
axs[0, 2].axis('square')
axs[0, 2].set_yticks([0, 10, 20, 30])
axs[0, 2].set_title('c) Sterrett & Bennett')
#axs[0, 2].set_xticks([0, 10, 20, 30, 40, 50, 60])
frames_sb = [data_sb[0], data_sb[1], data_sb[2], data_sb[3]]
maxindex_sb = identifymaxofmax(frames_sb)
axs[0, 2].plot(frames_sb[maxindex_sb], frames_sb[maxindex_sb], c = "grey", linewidth=0.5) #add 1:1 line behind
#sterrett and kelly
axs[1, 2].scatter(data_sk[0], data_sk[1], marker = "^", c = 'gold', label = "Prior")
axs[1, 2].scatter(data_sk[0], data_sk[2], marker = "x", c = 'k', label = "CaltoSelf")
axs[1, 2].scatter(data_sk[0], data_sk[3], marker = "s", c = 'blue', alpha=.5, label = "CaltoUpstream")
axs[1, 2].scatter(data_sk[0], data_sk[4], marker = "D", c = 'purple', alpha=.5, label = "CaltoDnstream")
axs[1, 2].axis('square')
axs[1, 2].set_title('f) Sterrett & Kelly')
axs[1, 2].set_xticks([0, 10, 20, 30, 40, 50])
axs[1, 2].set_xlabel("Obs peak flow (mgd)")  
frames_sk = [data_sk[0], data_sk[1], data_sk[2], data_sk[3]]
maxindex_sk = identifymaxofmax(frames_sk)
axs[1, 2].plot(frames_sk[maxindex_sk], frames_sk[maxindex_sk], c = "grey", linewidth=0.5) #add 1:1 line behind
#silver lake
axs[1, 0].scatter(data_sl[0], data_sl[1], marker = "^", c = 'gold')
axs[1, 0].scatter(data_sl[0], data_sl[2], marker = "x", c = 'k')
axs[1, 0].scatter(data_sl[0], data_sl[3], marker = "s", c = 'blue', alpha=.5)
axs[1, 0].scatter(data_sl[0], data_sl[4], marker = "D", c = 'purple', alpha=.5)
axs[1, 0].axis('square')
axs[1, 0].set_title('d) Silver Lake')
axs[1, 0].set_xlabel("Obs peak flow (mgd)") 
axs[1, 0].set_ylabel("Mod peak flow (mgd)")
frames_sl = [data_sl[0], data_sl[1], data_sl[2], data_sl[3]]
maxindex_sl = identifymaxofmax(frames_sl)
axs[1, 0].plot(frames_sl[maxindex_sl], frames_sl[maxindex_sl], c = "grey", linewidth=0.5) #add 1:1 line behind
#dunfermline and finance
axs[1, 1].scatter(data_df[0], data_df[1], marker = "^", c = 'gold')
axs[1, 1].scatter(data_df[0], data_df[2], marker = "x", c = 'k')
axs[1, 1].scatter(data_df[0], data_df[3], marker = "s", c = 'blue', alpha=.5)
axs[1, 1].scatter(data_df[0], data_df[4], marker = "D", c = 'purple', alpha=.5)
axs[1, 1].axis('square')
axs[1, 1].set_title('e) Dunfermline & Finance')
#axs[1, 1].set_xticks([0, 10, 20, 30, 40, 50])
axs[1, 1].set_xlabel("Obs peak flow (mgd)") 
frames_df = [data_df[0], data_df[1], data_df[2], data_df[3]]
maxindex_df = identifymaxofmax(frames_df)
axs[1, 1].plot(frames_df[maxindex_df], frames_df[maxindex_df], c = "grey", linewidth=0.5) #add 1:1 line behind
axs[1, 2].legend(bbox_to_anchor=(1.1, 1.05))
fig.tight_layout()
plt.savefig(path_to_save + "multiplotscatter.png")

#%% figures 4 S5 S6

#source of data for each of the multipanel line graphs
path_download = path#r"C:/Users/webbe/Downloads/"
filename_RMSE = "PCSWMM1D_RMSEresults.xlsx"
sheet_54 = "all results for 54 storms"
sheet_44 = "all results for 44 storms"
sheet_10 = "all results for 10 storms"
RMSE_54 = pd.read_excel(path_download + filename_RMSE, sheet_name = sheet_54)
RMSE_44 = pd.read_excel(path_download + filename_RMSE, sheet_name = sheet_44)
cols_54 = []
cols_44 = []
for i in range(len(RMSE_54.iloc[0])):
    if i > 0 and i <= 6:
        cols_54.append("RMSE_" + RMSE_54.iloc[0][i])
    elif i > 6 and i <= 12:
        cols_54.append(RMSE_54.iloc[0][i])
    elif i > 12 and i <= 22:
        cols_54.append("U_" + RMSE_54.iloc[0][i])
    else:
        cols_54.append(RMSE_54.iloc[0][i])
for i in range(len(RMSE_44.iloc[0])):
    if i > 0 and i <= 6:
        cols_44.append("RMSE_" + RMSE_44.iloc[0][i])
    elif i > 6 and i <= 12:
        cols_44.append(RMSE_44.iloc[0][i])
    elif i > 12 and i <= 22:
        cols_44.append("U_" + RMSE_44.iloc[0][i])
    else:
        cols_44.append(RMSE_44.iloc[0][i])
RMSE_54.columns = cols_54
RMSE_44.columns = cols_44
RMSE_54 = RMSE_54.drop([0], axis = 0).reset_index()
RMSE_44 = RMSE_44.drop([0], axis = 0).reset_index()
labellist = ['A003', 'B023', 'K027', 'P002', 'A009', 'D011']
bar_colors = ['blue', 'darkorange', 'grey', 'gold', 'deepskyblue', 'green']
addU = [0.44, 0.35, -1.19, 0.39, 0.45, -0.05]


multipanelplot6_scatterandsort(RMSE_44, 6, 12, 9, -3.5, 
                               ["Calibrated to: Kedron \n (A003)", "Homewood \n (B023)","Sterrett & Bennett \n (K027)",
                                "Sterrett & Kelly \n (P002)","Silver Lake \n (D011)","Dunfermline & Finance \n (A009)"],
                               [0.47, 0.58, -1.22, 0.81, 0.85, 0.45],
                               "mainfigure_6panel.png")


multipanelplot5_scatter(RMSE_44, 0, 5, 2, 0, 
                        ["Calibrated to: Prior", "equal weights", "SMARTER (smallest)", "SMARTER (largest"],
                        [0.77, 0.72, 0.69, 0.65, 0.66],
                        "mainfigure_5panel_6loc.png")

multipanelplot5_scatter(RMSE_44, 14, 19, 5, -2, 
                        ["Calibrated to: Prior", "equal weights", "SMARTER (smallest)", "SMARTER (largest"],
                        [0.77, 0.37, 0.75, -0.39, 0.65],
                        "mainfigure_5panel_4loc.png")