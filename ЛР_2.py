import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint


def SystemOfEquations(y, t):
    yt = np.zeros_like(y)
    yt[0] = y[2]
    yt[1] = y[3]

    a11 = (3 / 2) * m1 + m2
    a12 = m2 * h * np.cos(y[1])
    a21 = m2 * h * np.cos(y[1])
    a22 = J
    b1 = (M / R) * np.cos(omega * t) - c * y[0] + m2 * h * (y[3]) ** 2 * np.sin(y[1])
    b2 = -m2 * g * h * np.sin(y[1])

    yt[2] = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)
    yt[3] = (b2 * a11 - a21 * b1) / (a11 * a22 - a12 * a21)

    return yt

t = np.linspace(0, 20, 1001)  # массив времени

m1 = 40 #Масса колеса
m2 = 10 #Масса маятника
R = 0.1 #Радиус колеса
h = 0.06 #Расстояние до центра масс маятника
J = 0.04 #Момент инерции маятника
M = 39.2 #Момент силы
omega = np.pi #Угловая скорость
l0 = 2 #Пружина в нерастянутом расстояни
c = 100 #Жесткость пружины
g = 9.81

y0 = [0, 0.5, 0, 0]
#    [s,phi,Vs,Vphi]

sol = odeint(SystemOfEquations, y0, t)

s = sol[:,0]
phi = sol[:,1]
Vs = sol[:,2]
Vphi = sol[:,3]

Ws = np.array([SystemOfEquations(y, t)[2] for y,t in zip(sol,t)])
Wphi = np.array([SystemOfEquations(y, t)[3] for y,t in zip(sol,t)])

# График с рисунком
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-1, 5], ylim=[0, 6])


K = 22
Sh = 0.1 #ширина
b = 1/(K-2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K-1] = 1
Y_Spr[K-1] = 0
L_Spr = l0 + s #длина пружины


for i in range (K-2):
    X_Spr[i+1] = b*(i+1) - b/2
    Y_Spr[i+1] = Sh*(-1)**i

# Создание рисунка
Point1 = ax.plot(0, 3, marker = 'o', color='black')[0]  # точка крепежа
Drawed_Spring = ax.plot(X_Spr*L_Spr[0], 3 + Y_Spr,'black')[0]  # пружина
Point2 = ax.plot(l0 + s[0], 3, marker = 'o', color='black')[0]
Point0 = ax.plot(l0 + s[0] + R, 3, marker = 'o', color='black')[0]
Line20 = ax.plot([l0 + s[0], 3], [l0 + s[0] + R, 3], color='black')[0]
w = np.linspace(0, 2 * math.pi, 1001)
Circ = ax.plot(R * np.cos(w) + l0 + R + s[0], R * np.sin(w) + 3, 'black')[0]
PointA = ax.plot(l0 + s[0] + R + np.sin(phi[0]) * h, 3 - h * np.cos(phi[0]), marker = 'o', color='black')[0]
LineOA = ax.plot([l0 + s[0] + R, l0 + s[0] + R + np.cos(phi[0]) * h], [3, 3 - h * np.sin(phi[0])], color='black')[0]

LineY = ax.plot([0,0], [3-R,3+R*2], 'black')
LineX = ax.plot([0, 4*R + 2*l0], [3-R,3-R], 'black')

# анимация системы
def Kadr(i):

    Drawed_Spring.set_data(X_Spr * L_Spr[i], 3 + Y_Spr)
    Point2.set_data(l0 + s[i], 3)
    Point0.set_data(l0 + s[i] + R, 3)
    Line20.set_data([l0 + s[i], l0 + s[i] + R], [3, 3])
    Circ.set_data(R * np.cos(w) + l0 + R + s[i], R * np.sin(w) + 3)
    PointA.set_data(l0 + s[i] + R + np.sin(phi[i]) * h, 3 - h * np.cos(phi[i]))
    LineOA.set_data([l0 + s[i] + R, l0 + s[i] + R + np.sin(phi[i]) * h], [3, 3 - h * np.cos(phi[i])])
    return[Drawed_Spring]

# вызов функции анимации и демонстрация получившегося результата
anima = FuncAnimation(fig, Kadr, frames = len(t), interval = t[1]-t[0])
plt.show()
