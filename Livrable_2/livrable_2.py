import numpy as np
import matplotlib.pyplot as plt

def apprentissage_dictionnaire_surcomplet(X, taille_dictionnaire, nombre_iterations, alpha):
    """
    Fonction pour l'apprentissage d'un dictionnaire sur-complet.

    Args:
    - X: Matrice de données d'entraînement (chaque colonne représente une observation).
    - taille_dictionnaire: Taille du dictionnaire sur-complet.
    - nombre_iterations: Nombre d'itérations pour l'apprentissage.
    - alpha: Paramètre de régularisation.

    Returns:
    - D: Dictionnaire appris.
    """

    # Initialiser le dictionnaire de manière aléatoire
    D = np.random.rand(X.shape[0], taille_dictionnaire)

    for iteration in range(nombre_iterations):
        # Étape 1: Mise à jour des coefficients de représentation
        coefficients = np.linalg.lstsq(D, X, rcond=None)[0]

        # Étape 2: Mise à jour du dictionnaire
        for k in range(taille_dictionnaire):
            indices_non_zeros = np.where(coefficients[k, :] != 0)[0]
            if len(indices_non_zeros) == 0:
                continue

            Ek = X[:, indices_non_zeros] - np.dot(D, coefficients[:, indices_non_zeros])
            U, S, Vt = np.linalg.svd(Ek)

            # Mise à jour de la colonne du dictionnaire
            D[:, k] = U[:, 0]

        # Ajout d'une régularisation
        D /= np.linalg.norm(D, axis=0)

    # Affichage du nombre d'atomes après l'apprentissage
    print(f"Nombre d'atomes après l'apprentissage : {D.shape[1]}")

    return D

# Exemple d'utilisation
# Supposons que X soit votre matrice de données d'entraînement
X = np.random.rand(10, 100)
taille_dictionnaire = 20
nombre_iterations = 50
alpha = 0.1

# Apprentissage du dictionnaire
dictionnaire_appris = apprentissage_dictionnaire_surcomplet(X, taille_dictionnaire, nombre_iterations, alpha)
fig, ax = plt.subplots(1, taille_dictionnaire, figsize=(15, 3))

for k in range(taille_dictionnaire):
    ax[k].imshow(dictionnaire_appris[:, k].reshape((10, 1)), cmap='gray')
    ax[k].axis('off')
    ax[k].set_title(f'Atome {k+1}')

plt.show()
