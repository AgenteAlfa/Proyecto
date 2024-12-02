import numpy as np
import matplotlib.pyplot as plt


class Nodo:
      dic_dis = {}
      TAG = 0
      def __init__(self, x, y) -> None:
            self.Tag = Nodo.TAG
            Nodo.TAG += 1
            self.X = x
            self.Y = y
      def dis_to(self, nodo: 'Nodo'):
            proto_dis = Nodo.dic_dis.get((self.Tag, nodo.Tag), -1)
            if proto_dis == -1:
                  Nodo.dic_dis[(self.Tag, nodo.Tag)] = np.sqrt( (self.X - nodo.X)**2 + (self.Y - nodo.Y)**2 )
                  Nodo.dic_dis[(nodo.Tag, self.Tag)] = np.sqrt( (self.X - nodo.X)**2 + (self.Y - nodo.Y)**2 )
            return Nodo.dic_dis[(self.Tag, nodo.Tag)]
            
file = open("TPS_Data_152.txt", "r")
data_raw = [[int(item) for item in linea.replace("\n", "").split()] for linea in file.readlines()[1:]]
#data_raw = data_raw[5:30]
data = {d: Nodo(data_raw[d][1], data_raw[d][2])for d in range(len(data_raw))}
file.close()



N_GENERATIONS = 8000
MUTATION_RATE = 0.3
  #de 0 a 151
  
N_GENES = len(data)     #SON 152 elementos en data
X_BOUND = [0, len(data) - 1]

def F(X):
      res = []
      for Xi in X:
            distancia = 0
            for i in range(N_GENES - 2): #recorrer todas las ciudades
                  tag = Xi[i]
                  p_tag = Xi[i + 1]
                  distancia += data[tag].dis_to(data[p_tag])
            distancia += data[Xi[0]].dis_to(data[Xi[-1]]) #regresar a la primera ciudad
            res.append(distancia)
      return res
                  
POP_SIZE = 50
def mutar(child: np.ndarray):
      n1 = np.random.randint(len(child))
      n2 = np.random.randint(len(child) - n1 + 1)
      child[n1:n1 + n2] = np.flip(child[n1:n1 + n2], axis=0)
      return child 

raw = [mutar(mutar(list(range(len(data))))) for i in range(POP_SIZE)]
POP = np.array(raw)

def SeleccionLinealProb(rank, s = 1.5):
      return (2 - s)/ POP_SIZE + (2 * rank * (s - 1)) / (POP_SIZE * (POP_SIZE - 1))

def select_lineal(an):
      pop = np.array([i[0] for i in an])      
      idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=[SeleccionLinealProb(x) for x in range(POP_SIZE)])
      return pop[idx]

def crossover(padre, madre, cortes): 
      hijo = np.zeros_like(padre) - 1
      cross_points = [int(i*((cortes + 1)/N_GENES)) % 2 == 0 for i in range(N_GENES)] # Seleccion de los puntos de cruzamiento 
      hijo[cross_points] = padre[cross_points] # Apareamiento (Produccion de un hijo)     
      PV = [i for i in range(N_GENES) if i not in hijo]
      for i in range(N_GENES):
            if hijo[i] == -1:
                  if madre[i] in PV:
                        hijo[i] = madre[i]
                        PV.remove(madre[i])
                  else:
                        hijo[i] = PV[madre[i] % len(PV)]
                        PV.remove(PV[madre[i] % len(PV)])
                        
      return hijo   

def mutate(child: np.ndarray):
      if np.random.rand() < MUTATION_RATE:
            n1 = np.random.randint(len(child))
            n2 = np.random.randint(len(child) - n1 + 1)
            child[n1:n1 + n2] = np.flip(child[n1:n1 + n2], axis=0)
      return child  

lst_res = []
for iesima_generacion in range(N_GENERATIONS):
      # Se calcula el rendimiento de cada elemento y se junta a su poblacion para ordenarse y pasar a seleccionar
      #Para cada tipo de representacion   
      analisis = [list(a) for a in zip(POP, F(POP))]
      analisis.sort(key=lambda tup: tup[1], reverse=True)                          
      pop_padres = np.array([i[0] for i in analisis])       
      POP = select_lineal(analisis)
      
      hijos = np.zeros_like(POP)
      TOPS = POP[-int(POP_SIZE / 2):]
      i = 0
      for padre in TOPS:
            madre = POP[np.random.randint(POP_SIZE)]
            child1 = crossover(padre, madre,3)
            child1 = mutate(child1)
            
            child2 = crossover(madre, padre,3)
            child2 = mutate(child2)
            
            hijos[i] = child1
            hijos[i + int(POP_SIZE/2)] = child2
            i+=1
            
      POP = np.array(hijos)
      POP[-int(POP_SIZE / 3):] = pop_padres[-int(POP_SIZE / 3):]
      lst_res.append(analisis[-1][1])
resultados = analisis[-1]
    
inodos = resultados[0]
fig, ax = plt.subplots(figsize=(10, 4))
lstX = [data[i].X for i in inodos]
lstY = [data[i].Y for i in inodos]
for i in range(len(inodos)):
      ax.plot([lstX[i]],[lstY[i]], marker = "o")
for i in range(len(inodos) - 1):
      ax.plot([lstX[i],lstX[i + 1]],[lstY[i], lstY[i + 1]], marker = "")

print("Ruta mas corta ",resultados[1])

