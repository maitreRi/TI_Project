import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.linalg import svd

def apprentissage_dictionnaire_surcomplet(X, taille_dictionnaire, nombre_iterations, alpha):
    """
    Fonction pour l'apprentissage d'un dictionnaire sur-complet.

    Args:
    - X: Image d'entraînement (représentée en deux dimensions).
    - taille_dictionnaire: Taille du dictionnaire sur-complet.
    - nombre_iterations: Nombre d'itérations pour l'apprentissage.
    - alpha: Paramètre de régularisation.

    Returns:
    - D: Dictionnaire appris.
    """

    # Convertir l'image en une matrice de données d'entraînement
    X_mat = np.array(X).reshape(-1, 1)

    # Initialiser le dictionnaire de manière aléatoire
    D = np.random.rand(X_mat.shape[0], taille_dictionnaire)

    for iteration in range(nombre_iterations):
        # Étape 1: Mise à jour des coefficients de représentation
        coefficients = np.linalg.lstsq(D, X_mat, rcond=None)[0]

        # Étape 2: Mise à jour du dictionnaire
        for k in range(taille_dictionnaire):
            indices_non_zeros = np.where(coefficients[k, :] != 0)[0]
            if len(indices_non_zeros) == 0:
                continue

            Ek = X_mat[:, indices_non_zeros] - np.dot(D, coefficients[:, indices_non_zeros])
            
            # Utiliser la décomposition en valeurs singulières tronquée
            U, S, Vt = svd(Ek, full_matrices=False)

            # Mise à jour de la colonne du dictionnaire
            D[:, k] = U[:, 0]

        # Ajout d'une régularisation
        D /= np.linalg.norm(D, axis=0)

    return D

# Exemple d'utilisation avec une image JPEG
image_path = "2007041612_cam01p.jpg"
X = Image.open(image_path).convert("L")  # Convertir en niveaux de gris

taille_dictionnaire = 20
nombre_iterations = 50
alpha = 0.1

# Apprentissage du dictionnaire
dictionnaire_appris = apprentissage_dictionnaire_surcomplet(X, taille_dictionnaire, nombre_iterations, alpha)

# Affichage des atomes du dictionnaire appris
fig, ax = plt.subplots(1, taille_dictionnaire, figsize=(15, 3))

for k in range(taille_dictionnaire):
    ax[k].imshow(dictionnaire_appris[:, k].reshape(X.size[::-1]), cmap='gray')
    ax[k].axis('off')
    ax[k].set_title(f'Atome {k+1}')

plt.show()
