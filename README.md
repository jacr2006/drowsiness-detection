# Android TF Litemodel Drowsiness
1. Creacion de un modelo en TF Lite para deteccion de marcas faciales, usando el dataset (Helen)[https://ibug.doc.ic.ac.uk/resources/300-W/]<br/>
  1.1. Visualizacion de los datos->data_visualization.ipynb<br/>
  1.2. Procesamiento de los datos: escalado y normalizacion->data_processing.ipynb<br/>
  1.3. Entrenamiento: creacion de la CNN, compilacion y entrenamiento->data_training.ipynb<br/>
  1.4. Despliegue del modelo: pruebas del modelo->data_inference.ipynb<br/>
  1.5. Transformacion del modelo de H5 a Tensorflow Lite->keras_to_tflite_model.ipynb<br/><br/>
2. Despliegue del modelo TF Lite<br/><br/>
  2.1 Despliegue del modelo en PC con python->python_test_tflite_model.py<br/>
  ![alt test](Screenshot_Landmark.png)<br/>
  2.2 Despliegue del modelo en APP android<br/>
  ![alt test](Screenshot_Inference_android.jpg)<br/>
