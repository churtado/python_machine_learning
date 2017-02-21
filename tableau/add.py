import tabpy_client

client = tabpy_client.Client('http://localhost:9004/')

def add(x,y):
    import numpy as np
    return np.add(x, y).tolist()

client.deploy('add', add, 'Adds two numbers x and y',override=True)

x = [6.35, 6.40, 6.65, 8.60, 8.90, 9.00, 9.10]
y = [1.95, 1.95, 2.05, 3.05, 3.05, 3.10, 3.15]

print(client.query('add', x, y))
