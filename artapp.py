from matplotlib.image import imread
import os
import PIL
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import copy
import streamlit as st

st.sidebar.header("Acerca de")
st.sidebar.markdown("Esta Aplicación Web fue desarrollada por [Santiago Greloni](https://www.linkedin.com/in/santiago-greloni-4892a9196/).")
st.sidebar.markdown("En [este repositorio](https://github.com/SGreloni/segart/) se encuentra el \
                    código del proyecto en un Jupyter Notebook con algunos detalles adicionales.\
                    Ahí mismo también se puede encontrar el código con el que se realizó esta misma página")



st.title("SegArt")

st.markdown("En esta aplicación web busco mostrar una interesante utilización del *clustering*: la segmentación de imágenes.")
st.markdown("A partir del algoritmo *KMeans* se agrupan los pixeles en relación a su color, permitiendo segmentar la imagen\
            y creando composiciones minimalistas bastante atractivas.")

st.markdown("### ¿Qué es *KMeans*?")
st.markdown("*KMeans* es un algoritmo de aprendizaje no supervisado que permite encontrar una cantidad (K) de *clusters* en los que se agrupan los datos.")

cluster_img = PIL.Image.open("media_expl\clusters.png")
st.image(cluster_img, width = 350, caption = "Ejemplo de clusters en datos bidimensionales")

st.markdown("Este algoritmo funciona de la siguiente forma:")
st.markdown("1. Se eligen K puntos al azar que serán los centroides de los *clusters*")
st.markdown("2. Se asigna cada observación al cluster correspondiente al centroide más cercano")
st.markdown("3. Se actualiza la posición de los centroides como el promedio de la ubicación de las observaciones de cada *cluster*")
st.markdown("4. Se repiten el paso 2 y 3 hasta que el modelo converge (no cambia la posición de los centroides)")

st.markdown("Este proceso se puede visualizar en el siguiente video:")
cluster_img = PIL.Image.open("media_expl\kmeans.gif")
st.markdown("![Kmeans, paso por paso.](https://uploads.toptal.io/blog/image/92528/toptal-blog-image-1463672901961-c86610183bb2ba67f979c421f6748893.gif)")
st.markdown("Fuente: [Topal](https://www.toptal.com/machine-learning/clustering-algorithms)")

st.markdown("### Segmentación de imagen")


def leer_imagen(imagen):
    imagen = imread(os.path.join("imagenes", imagen), 0)
    imagen = imagen[:,:,:3]
    X = imagen.reshape(-1, 3)
    return imagen, X



if st.checkbox("Quiero subir mi propia imagen"):
    st.markdown("Por favor subir archivo en formato PNG. Si su imagen está en otro formato puede convertirlo en una pagina\
                como [esta](https://imagen.online-convert.com/es/convertir-a-png).")

    imagen = st.file_uploader(label = "Subir archivo", type= "png")
    if imagen is None:
        imagen, X = leer_imagen("pikachu.png")
    else:
        imagen = imread(imagen, 0)
        imagen = imagen[:,:,:3]
        X = imagen.reshape(-1, 3)



else:

    selecc_imagen = st.selectbox("Elija la imagen",
                                 ("pikachu", "pikachu2", "atardecer", "flor1", "flor2", "mario", "messi"))
    imagen, X = leer_imagen("{}.png".format(selecc_imagen))




k = st.slider("Elija K", 1, 10, 3)
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=128, random_state = 11).fit(X)
img_segmentada = kmeans.cluster_centers_[kmeans.labels_]
img_segmentada = img_segmentada.reshape(imagen.shape)




col1, col2 = st.beta_columns(2)

original = imagen
col1.header("Original")
col1.image(original, use_column_width=True, clamp = True)

segmentada = img_segmentada.astype('uint8')
col2.header("Segmentada")
col2.image(segmentada, use_column_width=True, clamp = True)

st.markdown("### Cambio de estilo")

kmeans_dup = copy.deepcopy(kmeans)

st.markdown("Como se puede ver, la segmentación da la oportunidad de editar las imágenes a una forma minimalista bastante estética.\
            Sin embargo esto no es todo, ahora que los colores son unos pocos, vale la pena intentar cambiarlos\
            para obtener resultados aún mejores y más personalizados.")

expansores = {}
selecc_color = {}
colores = kmeans_dup.cluster_centers_.astype(int)

def rgb_a_hex(lista_rgb):
    r, g, b = lista_rgb
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

for cluster in range(k):
    expansores[cluster] = st.beta_expander("Cambiar color {}".format(cluster +1))
    color_hex = rgb_a_hex(list(colores[cluster]))
    selecc_color[cluster] = expansores[cluster].color_picker('Seleccione un color', color_hex, key = "selecc_color {}".format(cluster))
    rgb = PIL.ImageColor.getcolor(selecc_color[cluster], "RGB")
    expansores[cluster].write("RGB: {}".format(rgb))
    colores[cluster] = rgb

img_cambiada = colores[kmeans_dup.labels_]
img_cambiada = img_cambiada.reshape(imagen.shape)

col1, col2 = st.beta_columns(2)



segmentada = img_segmentada.astype('uint8')
col1.header("Segmentada")
col1.image(segmentada, use_column_width=True, clamp = True)

cambiada = img_cambiada.astype('uint8')
col2.header("Cambiada")
col2.image(cambiada, use_column_width=True, clamp = True)


