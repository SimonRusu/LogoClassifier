# LogoClassifier

Se ha propuesto el desarrollo de una red que catalogue logos e indique que tipo de logo es (compras, transportes, personajes animados...). Para ello, se va a utilizar un dataset contenido en nuestra cuenta de Google Drive.


# **Configuración de modelo desarrollado:**

model = keras.Sequential()

model.add(Rescaling(scale=(1./127.5),
                    offset=-1, 
                    input_shape=(150, 150, 3)))

model.add(Conv2D(__8__, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(__8__, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(__32__, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

__model.add(Dropout(0.25))__

model.add(Conv2D(__64__, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

__model.add(Dropout(0.5))__

model.add(Conv2D(__128__, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

__model.add(Dropout(0.5))__

model.add(Flatten())

model.add(Dense(__32__, activation='__sigmoid__'))

__model.add(Dropout(0.25))__

model.add(Dense(15, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])

# Razonamiento:
Tras las numerosas pruebas realizadas con el modelo de la red neuronal se ha concluido lo siguiente:
* Es preferible que las primeras capas del modelo posean pocas neuronas para ir expandiendo las capas posteriores y finalmente converger en un menor número de neuronas.
* Si bien el modelo de activación relu ocasiona que la red aprenda de manera mas fluida y rápida se ha decidido utilizar el modelo de activación sigmoide en la penultima capa. Esto es debido a una mejor tasa de acierto cuando se utilizan imagenes con variantes singificativas respecto a las utilizadas en el entrenamiento.
* La importancia de la adicion de Dropouts en las capas finales del aprendizaje (a mayor número de neuronas mayor dropout y viceversa). Un problema común era la obtención de una tasa de acierto muy elevada (entorno al 97%) pero dichos resultados de poco servian ya que la red neuronal no solia acertar cuando se le aportaban nuevas imagenes no incluidas en el dataset de entrenamiento. Lo que se ha conseguido con las capas de dropout es reducir el sobreajuste y ajustar la tasa de acierto a valores mas proximos a los esperados, además, se obtiene una red mucho mas robusta frente a nuevas imagenes no incluidas en el dataset de entrenamiento.

Se han realizado 3 entrenamientos a la red desarrollada y 1 entrenamiento a la red VGG16.

__- ENTRENAMIENTO 1__

![Entrenamiento 1](https://user-images.githubusercontent.com/91427107/147878963-e1015a4a-b386-40b5-bff1-04bd20f5e388.png)
![grafica red desarrollada](https://user-images.githubusercontent.com/91427107/147878961-29f9d308-6737-46a2-9a2c-60c186c99e17.png)

__- ENTRENAMIENTO 2__

![Entrenamiento 2](https://user-images.githubusercontent.com/91427107/147878959-5aff027d-cf56-4d62-b1e4-fe62868c3493.png)
![grafica red desarrollada 2](https://user-images.githubusercontent.com/91427107/147878960-5caab85a-d629-4bb6-af93-2ec472f726ab.png)

__- ENTRENAMIENTO 3__

![Entrenamiento 3](https://user-images.githubusercontent.com/91427107/147878955-0369d80e-965f-4597-bb54-e57690b0be6d.png)
![grafica red desarrollada 3](https://user-images.githubusercontent.com/91427107/147878956-c0e04a68-1bf5-451a-b8cf-999533b8bb8d.png)

__- ENTRENAMIENTO VGG16__

![Entrenamiento VGG16](https://user-images.githubusercontent.com/91427107/147878953-05d22e79-6df4-420e-94c8-2c892ab3ff78.png)
![grafica red VGG16](https://user-images.githubusercontent.com/91427107/147879048-9519e5c5-f175-4744-9b97-8773d5cb5ae7.png)

Tal y como podemos observar la red neuronal desarrollada va tomando mejores resultados a medida que avanza el entrenamiento.
Adicionalmente se han comparado los resultados obtenidos con la red VGG16 incluida en keras, dicha red ya esta preetrenada y obtiene unos resultados excelentes por lo que es ideal para comprobar si los resultados obtenidos coinciden con lo esperado.

# Conclusion:

La red VGG16 ha logrado obtener una tasa de acierto que oscila entre el __92%-95%__ con un solo entrenamiento, mientras que la red desarrollada se mantiene en una tasa del __90%-93%__ tras múltiples entrenamientos. Se puede concluir que los resultados esperados son satisfactorios debido a que se han realizado múltiples pruebas con imágenes no incluidas en el dataset y se obtienen resultados similares. Por último, junto al comprimido con los dataset se ha incluido una carpeta llamada __test__ con un conjunto de imágenes para verificar el correcto funcionamiento de ambas redes.
