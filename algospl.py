import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.signal import wiener

def image_blocking(image, block_size):
    # Obtenir les dimensions de l'image
    height, width = image.shape
    
    # Calculer le nombre de blocs en hauteur et en largeur
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size
    
    # Redimensionner l'image pour s'assurer qu'elle est divisible par block_size
    image = image[:num_blocks_height * block_size, :num_blocks_width * block_size]
    
    # Diviser l'image en blocs
    blocks = image.reshape((num_blocks_height, block_size, num_blocks_width, block_size))
    
    # Permuter les dimensions pour obtenir une liste de blocs
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    
    return blocks, num_blocks_height, num_blocks_width

def BCS_SPL_DCT_iteration(y, Phi, D, x_k, itermax=200, tolerance=1e-6, lambda_=6):
    for _ in range(itermax):
        # Calculer le résidu
        residual = y - Phi @ D @ x_k
        
        # Calculer le gradient
        gradient = Phi.T @ D.T @ residual
        
        # Mettre à jour x_k
        x_k = x_k + gradient
        
        # Seuillage
        alpha_k1 = np.abs(D.T @ x_k)
        threshold = lambda_ * np.median(alpha_k1) * 0.6745 / np.sqrt(2 * np.log(x_k.shape[0] * x_k.shape[1]))
        alpha_k1 = np.where(alpha_k1 > threshold, alpha_k1, 0)
        
        # Reconstruire le vecteur x
        x_k = D @ alpha_k1
        
        # Mettre à jour la solution
        x_k = x_k + Phi.T @ D.T @ residual
        
        # Appliquer le filtre de Wiener
        x_k = wiener(x_k.reshape(D.shape[1], -1), (3, 3))
        x_k = x_k.reshape((-1, 1))
        
        # Vérifier la convergence
        if np.linalg.norm(gradient) < tolerance:
            break
    
    return x_k

def BCS_SPL_DCT(y, Phi, D, num_blocks_height, num_blocks_width, block_size, itermax=200, tolerance=1e-6, lambda_=6):
    x_hat = np.zeros((Phi.shape[1], num_blocks_height * num_blocks_width * block_size**2))
    
    for j in range(num_blocks_height * num_blocks_width):
        y_j = y[:, j * block_size**2 : (j + 1) * block_size**2].reshape((-1, 1))
        x_j = BCS_SPL_DCT_iteration(y_j, Phi, D, x_hat[:, j * block_size**2 : (j + 1) * block_size**2], itermax, tolerance, lambda_)
        x_hat[:, j * block_size**2 : (j + 1) * block_size**2] = x_j.flatten()
    
    # Remonter les blocs pour obtenir l'image complète
    x_hat = x_hat.reshape((Phi.shape[1], num_blocks_height, num_blocks_width, block_size, block_size))
    x_hat = x_hat.transpose(0, 2, 1, 3, 4).reshape((Phi.shape[1], num_blocks_height * block_size, num_blocks_width * block_size))
    
    return x_hat

# Charger l'image
image = cv2.imread('2007041608_cam01p.jpg', cv2.IMREAD_GRAYSCALE)

# Taille des blocs pour la DCT2D
block_size = 8

# Diviser l'image en blocs
blocks, num_blocks_height, num_blocks_width = image_blocking(image, block_size)

# Générer la matrice Phi en échantillonnant aléatoirement des lignes de la matrice identité
num_measurements = 100
Phi = np.eye(block_size**2)[:num_measurements, :]

# Générer la matrice de transformation D avec DCT2D
D = np.zeros((block_size**2, block_size**2))
for u in range(block_size):
    for v in range(block_size):
        D[u * block_size + v, :] = np.cos((2 * (u if u > 0 else 1) + 1) * np.pi * np.arange(block_size) * v / (2 * block_size))

# Normaliser D
D /= np.sqrt(2 / block_size**2)

# Générer un vecteur y pour tester l'algorithme
y = Phi @ blocks.reshape((-1, block_size**2)).T

# Nombre de blocs
NB = num_blocks_height * num_blocks_width

# Exemple d'utilisation de l'algorithme
reconstructed_image = BCS_SPL_DCT(y, Phi, D, num_blocks_height, num_blocks_width, block_size)
cv2.imshow('Image Reconstruite', reconstructed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
