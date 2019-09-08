clc;
clear all;
filenamere = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/RE/Caracateristicas_RE_H(S)V_50x50.xlsx';
RE = xlsread(filenamere);
filenamedm = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/DM/Caracateristicas_DM_H(S)V_50x50.xlsx';
DM = xlsread(filenamedm);
filenameno = 'C:/Users/Nataly/Documents/Trabajo-de-grado_Artefactos/balanceoDM/50x50/NO/Caracateristicas_NO_H(S)V_50x50.xlsx';
NO = xlsread(filenameno);



statDRg=zeros(102,1);
pDRg=zeros(102,1);
statDNg=zeros(102,1);
pDNg=zeros(102,1);
statRNg=zeros(102,1);
pRNg=zeros(102,1);


for z=2:1:103  
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
nueva(:,1)=pDRg;
nueva(:,2)=pDNg;
nueva(:,3)=pRNg;
nueva(:,4)=statDR;
nueva(:,5)=statDN;
nueva(:,6)=statRN;
xlswrite('WILCONXON_H(S)V_matlab_50x50.xls', nueva);
   


