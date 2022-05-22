Con respecto a la nueva versión:

La hucha y la maceta lsa detecta bien con los keypoints, el problema es que la taza y el PLC no. Existen varios posibles candidatos al problema:

1. Keypoints: El número de keypoints puede ser demasiado pequeño e insuficiente y por eso falla. El caso es que se ha probado a realziar el 
   cálculo de descriptores usando la nube de puntos voxelizada (lo cual ya es un número de puntos considerable para que pueda encontrarlos)
   y sigue dado una mala transformación, por lo cual pienso que esta opción se puede descartar.
   
2. Estimación de las normales: Puede que alguno de los parámetros no concuerde, lo cual me parece menos probable.

3. Global Registration: Como utiliza RANSAC y apara dascartar los malos emparejamientos utiliza los llamados "checkers" para comprobarlo tales
   como la distanica entre bordes o la distancia tal cual entre puntos (threshold), entiendo que alguno de eesos parámetros puede fallar. También
   puede ser por el número de iteraciones o el grado de confianza (probabilidad), pero en la documentación pone que estos son los parámetros por
   defecto y en el paper que se han calculado empíricamente.
   
La transformación es incorrecta y tiene toda la pinta de ser por culpa del global registration, porque luego hay que hacer ICP que es la local 
registration y para eso se necesita una transformación inicial, que es la obtenida con RANSAC, para refinarla. Con el global nos da igual la 
posición y orientación del objeto en la escena, por eso con ICPlo que hacemos es un refinamiento de esa transformación ya considerando posición
y orientación, pero si con RANSAC la transformación ya sale muy alejada, ICP no puede hacer maravillas. 

Además, lo que no me cuadra es que con los dos objetos anteriores de perfecto pero con estos dos no.
