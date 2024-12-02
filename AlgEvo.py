import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import enum

ITERACIONES = 20
N_CORTES = 1

"""class Nodo:
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
      def Reiniciar():
            Nodo.dic_dis.clear()
            TAG = 0
"""
class AE:
      def __init__(self, cantidad_genes, funcion, iteraciones = 1, pop_inicial = 30, ngeneraciones = 120) -> None:
            self.F = funcion
            self.N_GENES = cantidad_genes
            self.MAXIMO = cantidad_genes - 1#0 - NG-1
            self.POP_SIZE = pop_inicial
            self.MUTATION_RATE = 0.3
            self.N_GENERATIONS = ngeneraciones
            self.ITERACIONES = iteraciones
            self.resultados = []
            self.GenerarPopInicial()

      def GenerarPopInicial(self):
            
            raw = [self.Nmutate(12, list(range(self.N_GENES))) for i in range(self.POP_SIZE)]
            self.POP = np.array(raw)

      def Nmutate(self, n, child: np.ndarray):
            for i in range(n):
                  child = self.mutate(child)
            return child
      def mutate(self, child: np.ndarray):
            if np.random.rand() < self.MUTATION_RATE:
                  n1 = np.random.randint(len(child))
                  n2 = np.random.randint(len(child) - n1 + 1)
                  child[n1:n1 + n2] = np.flip(child[n1:n1 + n2], axis=0)
            return child  
      
      def SeleccionLinealProb(self, rank, s = 1.5):
            return (2 - s)/ self.POP_SIZE + (2 * rank * (s - 1)) / (self.POP_SIZE * (self.POP_SIZE - 1))

      def crossover(self, padre, madre, cortes): 
            hijo = np.zeros_like(padre) - 1
            cross_points = [int(i*((cortes + 1)/self.N_GENES)) % 2 == 0 for i in range(self.N_GENES)] # Seleccion de los puntos de cruzamiento 
            hijo[cross_points] = padre[cross_points] # Apareamiento (Produccion de un hijo)     
            PV = [i for i in range(self.N_GENES) if i not in hijo]
            for i in range(self.N_GENES):
                  if hijo[i] == -1:
                        if madre[i] in PV:
                              hijo[i] = madre[i]
                              PV.remove(madre[i])
                        else:
                              hijo[i] = PV[madre[i] % len(PV)]
                              PV.remove(PV[madre[i] % len(PV)])       
            return hijo
      def select_lineal(self,an):
            pop = np.array([i[0] for i in an])      
            idx = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True,p=[self.SeleccionLinealProb(x) for x in range(self.POP_SIZE)])
            return pop[idx]
      def Entrenar(self):
            for iteracion in range(self.ITERACIONES):
                  for iesima_generacion in range(self.N_GENERATIONS):
                        # Se calcula el rendimiento de cada elemento y se junta a su poblacion para ordenarse y pasar a seleccionar
                        #Para cada tipo de representacion   
                        analisis = [list(a) for a in zip(self.POP, self.F(self.POP))]
                        analisis.sort(key=lambda tup: tup[1], reverse=True)                          
                        pop_padres = np.array([i[0] for i in analisis])       
                        self.POP = self.select_lineal(analisis)
                        
                        hijos = np.zeros_like(self.POP)
                        TOPS = self.POP[-int(self.POP_SIZE / 2):]
                        i = 0
                        for padre in TOPS:
                              madre = self.POP[np.random.randint(self.POP_SIZE)]
                              child1 = self.crossover(padre, madre,3)
                              child1 = self.mutate(child1)
                              
                              child2 = self.crossover(madre, padre,3)
                              child2 = self.mutate(child2)
                              
                              hijos[i] = child1
                              hijos[i + int(self.POP_SIZE/2)] = child2
                              i+=1
                              
                        self.POP = np.array(hijos)
                        self.POP[-int(self.POP_SIZE / 3):] = pop_padres[-int(self.POP_SIZE / 3):]
                        self.resultados.append([iesima_generacion, analisis[-1][1]])
                  resultados = analisis[-1]
            return resultados

      def GraficarResultados(self):
            df = pd.DataFrame(self.resultados, columns=["Generacion","Resultado"])
            res = df.groupby(["Generacion"]).sum() / self.ITERACIONES
            res = res.reset_index()
            ax_legends = []
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_xlabel('Generacion')
            ax.set_ylabel('Resultados')
            ax.set_title("Generacion vs Resultados")

            sc = pd.Series(res.Resultado.values, index=res.Generacion,name='Generacion vs Resultados')
            ax = sc.plot(ax=ax, style='--', linewidth=2)
            ax_legends.append(f"Promedio de mejores individuo en cada generaciones")
            ax.legend(ax_legends)


class AlgEvo:
      CROSS_RATE = 0.8
      MUTATION_RATE = 0.01
      N_GENERATIONS = 30
      X_BOUND = [-40, 40] #Rango de los valores
      POP_SIZE = 20     #10 individuos 
      DNA_SIZE = 12      #Nos permite 2^6 valores posibles que se generaran 4 veces
      N_GENES = 2
      VERBOSE = False
      PADRES = False
      POP = np.random.randint(2, size=(POP_SIZE, N_GENES , DNA_SIZE))   #Se usara la representacion en GRAY
      lst_res = []
      def RegenerarPOP(): 
            AlgEvo.POP = np.random.randint(2, size=(AlgEvo.POP_SIZE, AlgEvo.N_GENES , AlgEvo.DNA_SIZE)) 

      def F(X):
            x = X[0]
            y = X[1]
            return abs(x + y)

      def crossover(padre, madre, cortes): 
            hijo = padre.copy()
            cross_points = [int(i*((cortes + 1)/AlgEvo.DNA_SIZE)) % 2 == 0 for i in range(AlgEvo.DNA_SIZE)] # Seleccion de los puntos de cruzamiento 
            for i in range(AlgEvo.N_GENES):
                  hijo[i,cross_points] = madre[i, cross_points] # Apareamiento (Produccion de un hijo)                  
            return hijo    

      def mutate(child):
            for i_crom in range(AlgEvo.N_GENES):
                  for point in range(AlgEvo.DNA_SIZE):
                        if np.random.rand() < AlgEvo.MUTATION_RATE:
                              child[i_crom,point] = 1 if child[i_crom,point] == 0 else 0
            return child   
      
      def SeleccionLinealProb(rank, s = 1.5):
            return (2 - s)/ AlgEvo.POP_SIZE + (2 * rank * (s - 1)) / (AlgEvo.POP_SIZE * (AlgEvo.POP_SIZE - 1))
      
      def select_lineal(an, VERBOSE = False):
            pop = np.array([i[0] for i in an])      
            idx = np.random.choice(np.arange(AlgEvo.POP_SIZE), size=AlgEvo.POP_SIZE, replace=True,p=[AlgEvo.SeleccionLinealProb(x) for x in range(AlgEvo.POP_SIZE)])
            if(VERBOSE): print("Padres :", idx)
            return pop[idx]

      def GrayGenToDec(xg):
            DNA_SIZE = AlgEvo.DNA_SIZE
            X_BOUND = AlgEvo.X_BOUND
            xd_g = xg.dot(2 ** np.arange(DNA_SIZE)[::-1])
            xd = []
            for r in xd_g:
                  res = r
                  while r > 0:
                        r >>= 1
                        res ^= r
                  res = res / float(2**DNA_SIZE-1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
                  xd.append(int(res))
            return np.array(xd)
      def GrayToDec(xg):
            return np.array([AlgEvo.GrayGenToDec(i) for i in xg])

      def IterarGraficar():
            dic_resIteracion = {}
            AlgEvo.RegenerarPOP()
            for iteracion in range(ITERACIONES):
                  for iesima_generacion in range(AlgEvo.N_GENERATIONS):
                        #print(f"Cortes : {n_cortes} - Iteracion : {iteracion} - Generacion : {iesima_generacion}")
                        # Se calcula el rendimiento de cada elemento y se junta a su poblacion para ordenarse y pasar a seleccionar
                        #Para cada tipo de representacion
                        F_values = AlgEvo.F(AlgEvo.GrayToDec(AlgEvo.POP).T)
                        fitness = F_values
                        analisis = [list(a) for a in zip(AlgEvo.POP, fitness)]
                        analisis.sort(key=lambda tup: tup[1], reverse=True)
                        AlgEvo.lst_res.append([iesima_generacion, analisis[-1][1]])                              
                        pop_padres = np.array([i[0] for i in analisis]) 
                        AlgEvo.POP = AlgEvo.select_lineal(analisis, VERBOSE = False)
                        hijos = []
                        for parent in AlgEvo.POP:
                              if np.random.rand() < AlgEvo.CROSS_RATE:
                                    madre = AlgEvo.POP[np.random.randint(AlgEvo.POP_SIZE)]
                                    child = AlgEvo.crossover(parent, madre, N_CORTES)
                              else:
                                    child = parent.copy()
                              child = AlgEvo.mutate(child)
                              hijos.append(child)
                        AlgEvo.POP = np.array(hijos)
                        if (AlgEvo.PADRES):
                              AlgEvo.POP[-2:] = pop_padres[-2:]
                  dic_resIteracion[iteracion] = pop_padres[-1]
                  AlgEvo.RegenerarPOP()
            df = pd.DataFrame(AlgEvo.lst_res, columns=["Generacion","Resultado"])
            #df['Numero de cortes'] = df['Numero de cortes'].astype(int)
            res = df.groupby(["Generacion"]).sum() / ITERACIONES
            res = res.reset_index()
            ax_legends = []
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_xlabel('Generacion')
            ax.set_ylabel('Resultados')
            ax.set_title("Generacion vs Resultados")
      
            sc = pd.Series(res.Resultado.values, index=res.Generacion,name='Generacion vs Resultados')
            ax = sc.plot(ax=ax, style='--', linewidth=2)
            ax_legends.append(f"Promedio de mejores individuo en cada generaciones")
            ax.legend(ax_legends)
            return dic_resIteracion
      