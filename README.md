# TFG
# Análisis e implementación de sistemas de comunicaciones mediante Autoencoders profundos.

Se presentan y discuten las ventajas e inconvenientes que introduce el uso de Autoencoders, un tipo de red neuronal, en un sistema de comunicaciones ruidoso. Como contraposición, se realiza la comparativa con la aplicación de técnicas de codificación convencionales, en concreto los códigos Hamming. Se proporcionarán la arquitectura y el diseño en código programado de las redes neuronales planteados para cada caso. Puede verse que las prestaciones del Autoencoder se aproximan enormemente a las de la detección MLD para el caso de un canal AWGN. También se estudia el desempeño del Autoencoder en un sistema con interferencia entre símbolos, para el que el ruido ya no será blanco y dependeremos del uso de ecualizadores. Se le exigirá además aprender a ecualizar todo el canal, sin conocimiento previo del mismo ni compensación de ISI por ecualización forzada. Los resultados indicarán que este tipo de redes neuronales efectivamente es capaz de aprender a detectar los símbolos con interferencia y a trabajar con canales no AWGN.

Para ejecutar las simulaciones debe utilizarse el script main.py. En él se eligen tanto el Autoencoder a usar en la simulación (ruido AWGN o distintos tipos de ecualización) como las curvas de codificación convencionales para comparar los resultados. 
Las configuraciones de entrenamiento del autoencoder se modifican en los scripts dentro de la carpeta /autoencoder
