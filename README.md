# Blocked Compressed Sensing Image Reconstruction.
Achile Masson, Romaric Vandycke, Thomas Faguet.

Professeur Reférent : Nisrine Fortin

Ce projet a pour objectif d'utliser le compressive sensing pour eliminer les perturbation des images que nous recevons en entrée et reduire la taille des données de celle-ci.
Pour cela nous allon utiliser l'acquisition comprimée afin de mesurer et comprimer l'image en même temps. Le principe recompose sur la construction d'un dictionaire contenant les données de notre image et de telles sortes que nous pouvons reconstruire n'importe quelle vecteur de notre image en utilisant un vecteur colone le plus parcimonieux possible ie contenant le plus de zéros possible.

#Livrable 1:

Dans cette premiére partie du projet on s'intéresse aux pré-traitement de l'image, l'extraction de batch de l'image et leur vectorisation ainsi que la reconstruction  de l'image.
Le rendu est composée de quatre fonctions que je vais détailer.

ExtractPacth:

La fonction a pour but de decouper l'image en plus petite portin de celle-ci, pour cela on va utiliser la méthode crop qui va nous renvoyer une region rectangulaire de l'image.

Patch2Vector:

Cette fonction va prendre en entré la portion d'image recu precedamment pour ensuite la transformer en tableau avec la fonction numpy.array.
Finalement on la reduit à une dimension avec la methode flatten.

Vector2Pacth:

Ici, nous faison la méthode inverse. On utilise la fonction reshape pour pour transformer le vecteur colone en matrice de taille approprier, puis on le transforme en objet Image avec la fonction Image.fromarray.

ReconstImage:

Afin de reconstruire l'image on va créer une nouvelle image avec la Image.new que nous allon ustiliser pour poser par dessus les patch que nous avons reconstruit précedemant avec la méthode paste.
