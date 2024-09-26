import numpy as np
import matplotlib.pyplot as plt

# Definir una matriz A mal condicionada y un vector b con diferentes valores
A = np.array([[0.005, 1, 2], 
              [1, 0.005, 3], 
              [2, 3, 0.005]])

b = np.array([2, 3, 5])

# Calcular el determinante de la matriz A
determinante = np.linalg.det(A)
print(f'Determinante de A: {determinante:.5e}')

# Intentar calcular la inversa de la matriz (si es posible)
try:
    A_inv = np.linalg.inv(A)
    print('Matriz Inversa de A:\n', A_inv)
except np.linalg.LinAlgError:
    print("La matriz es singular y no tiene inversa.")

# Calcular el número de condición
condicional = np.linalg.cond(A)
print(f'Número de condición de A: {condicional:.5e}')

# Resolver el sistema original Ax = b
solucion_original = np.linalg.solve(A, b)
print('Solución del sistema original:', solucion_original)

# Hacer una pequeña perturbación en el vector b
b_perturbado = b + np.array([0.01, -0.02, 0.03])

# Resolver el sistema con el vector b perturbado
solucion_perturbada = np.linalg.solve(A, b_perturbado)
print('Solución del sistema perturbado:', solucion_perturbada)

# Comparar gráficamente las soluciones original y perturbada
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos de la solución original y perturbada
ax.scatter(solucion_original[0], solucion_original[1], solucion_original[2], color='blue', s=100, label='Solución Original')
ax.scatter(solucion_perturbada[0], solucion_perturbada[1], solucion_perturbada[2], color='red', s=100, label='Solución Perturbada')

# Configurar etiquetas y título
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_zlabel('x3', fontsize=12)
plt.title('Comparación de Soluciones (Original vs Perturbada)', fontsize=14)

# Mejorar estética del gráfico
ax.legend(loc='upper left', fontsize=12)
ax.grid(True)
plt.show()

# Comparación de la diferencia entre las soluciones
diferencia = np.linalg.norm(solucion_original - solucion_perturbada)
print(f'Diferencia entre soluciones: {diferencia:.5e}')
