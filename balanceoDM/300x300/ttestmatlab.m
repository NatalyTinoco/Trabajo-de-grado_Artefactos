clc;
clear all;
% filenamere = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/RE/Caracateristicas_RE_H(S)V_50x50.xlsx';
% RE = xlsread(filenamere);
% filenamedm = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/DM/Caracateristicas_DM_H(S)V_50x50.xlsx';
% DM = xlsread(filenamedm);
% filenameno = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/NO/Caracateristicas_NO_H(S)V_50x50.xlsx';
% NO = xlsread(filenameno);

filenamere = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/RE/Caracateristicas_RE_H(S)V_300x300.xlsx';
RE = xlsread(filenamere);
filenamedm = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/DM/Caracateristicas_DM_H(S)V_300x300.xlsx';
DM = xlsread(filenamedm);
filenameno = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/500x500/NO/Caracateristicas_NO_H(S)V_300x300.xlsx';
NO = xlsread(filenameno);



statDRg=zeros(102,1);
pDRg=zeros(102,1);
statDNg=zeros(102,1);
pDNg=zeros(102,1);
statRNg=zeros(102,1);
pRNg=zeros(102,1);


for z=1:103  
    %[statDR,pDR]=ttest2(DM(:,z),RE(:,z));
    [statDR,pDR]=ranksum(DM(:,z),RE(:,z));
    statDRg(z,1)=statDR;
    pDRg(z,1)=pDR;
    
    %[statDN,pDN]=ttest2(DM(:,z),NO(:,z));
    [statDN,pDN]=ranksum(DM(:,z),NO(:,z));
    statDNg(z,1)=statDN;
    pDNg(z,1)=pDN;
    
    %[statRN,pRN]=ttest2(RE(:,z),NO(:,z));
    [statRN,pRN]=ranksum(RE(:,z),NO(:,z));
    statRNg(z,1)=statRN;
    pRNg(z,1)=pRN;
end
nueva=zeros(length(pRNg),6);
nueva(:,6)=pDRg;
nueva(:,5)=pDNg;
nueva(:,4)=pRNg;
nueva(:,3)=statDR;
nueva(:,2)=statDN;
nueva(:,1)=statRN;
xlswrite('WILCONXON_H(S)V_matlab_300x300.xls', nueva);
   


