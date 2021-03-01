import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

class linear:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.b_0 = self.b_1 = 0

  def fit(self):
     n = np.size(self.x)
     m_x, m_y = np.mean(self.x), np.mean(self.y)
     error_xy = np.sum(self.y*self.x) - n*m_y*m_x
     error_xx = np.sum(self.x*self.x) - n*m_x*m_x

     self.b_1 = error_xy / error_xx
     self.b_0 = m_y - self.b_1*m_x
  
  def fit_with_gradient(self, lr, thresh, iters):
    n = np.size(self.x)
    curr_b_0 = curr_b_1 = 0
    for i in range(iters):
      pred_y = self.x*curr_b_0 + curr_b_1
      d_m = (-2/n) * np.sum(self.x * (self.y - pred_y))
      d_c = (-2/n) * np.sum(self.y - pred_y)
      if np.all(np.abs(d_m) + np.abs(d_c) <= thresh):
            print("threshold of", thresh, "achieved at iter", i)
            break
      curr_b_0 = curr_b_0 - lr * d_m
      curr_b_1 = curr_b_1 - lr * d_c 
    self.b_1 = curr_b_0
    self.b_0 = curr_b_1

  def plot(self):
    plt.scatter(self.x, self.y)

    y_pred = self.b_0 + self.b_1*self.x
    
    plt.plot(self.x, y_pred)
    plt.show()

  def predict(self, test_x):
    return self.b_0 + self.b_1 * test_x

def main():
  data = pd.read_csv("01-food-profit.csv")
  x, y = data.iloc[1:, 0], data.iloc[1:, 1]

  print("Fitting with gradient descent\n-----------------------------------------------")
  lin = linear(x, y)
  lin.fit_with_gradient(.01, .005, 5000)
  print(lin.b_0, lin.b_1)
  lin.plot()

  print("Fitting\n-----------------------------------------------")
  lin.fit()
  print(lin.b_0, lin.b_1)
  lin.plot()

if __name__ == "__main__": 
	main() 
