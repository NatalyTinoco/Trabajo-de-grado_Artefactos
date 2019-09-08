# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 21:16:30 2019

@author: Nataly
"""

import pandas as pd 
import pymrmr 
df = pd.read_excel ('numeroCa.xls') 

xl = pd.ExcelFile('foo.xls')

xl.sheet_names  # see all sheet names

xl.parse(sheet_name) 
#pymrmr.mRMR (df, 'MIQ', 10) 