import threading
import random
import time
N = 300
THREADS = 4
A = [[random.random() for _ in range(N)] for _ in range(N)]
B = [[random.random() for _ in range(N)] for _ in range(N)]
C = [[0]*N for _ in range(N)]
def calcular(inicio, fim):
    for i in range(inicio, fim):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
inicio = time.time()
threads = []
linhas = N // THREADS
for t in range(THREADS):
 ini = t * linhas
 fim = N if t == THREADS-1 else (t+1)*linhas
 th = threading.Thread(target=calcular, args=(ini, fim))
 threads.append(th)
 th.start()
for th in threads:
 th.join()
fim = time.time()
print("Tempo com threads:", (fim-inicio)*1000, "ms")