clc;
clear all;
% filenamere = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/RE/Caracateristicas_RE_H(S)V_50x50.xlsx';
% RE = xlsread(filenamere);
% filenamedm = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/DM/Caracateristicas_DM_H(S)V_50x50.xlsx';
% DM = xlsread(filenamedm);
% filenameno = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/NO/Caracateristicas_NO_H(S)V_50x50.xlsx';
% NO = xlsread(filenameno);

filenamere = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/numeroCa.xlsx';
d = xlsread(filenamere);
filenamedm = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/clases.xlsx';
f= xlsread(filenamedm);
K=10;
fea=mrmr_mid_d(d, f, K);
