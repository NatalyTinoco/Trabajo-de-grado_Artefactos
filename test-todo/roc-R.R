library("readxl")
library(pROC)
my_data <- read_excel("C:/Users/Usuario/Documents/Daniela/Tesis/Trabajo-de-grado_Artefactos/test-todo/predicciones_DM.xlsx")
y_test <- my_data['y_test']
y_pred <- my_data['y_pred']

prueba <- t(y_test)
prueba2 <- t(y_pred)

roc(prueba, prueba2, plot=TRUE)


