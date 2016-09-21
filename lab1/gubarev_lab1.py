__author__ = 'yang'

import random
import numpy
import math
import matplotlib.pyplot as plt
from openpyxl import Workbook


CONST_SHOW_PLOTS = False

CONST_WRITE_SVD = False

# random complex pairs constants
CONST_SIZE = 5
CONST_FUNC_SIZE = CONST_SIZE * 3
CONST_FIRST_ALIGN = 5.0
CONST_FIRST_MULTIPLIER = 2.0
CONST_SECOND_ALIGN = 5.0
CONST_SECOND_MULTIPLIER = 2.0
CONST_DIVISOR = 1e4
CONST_T_X_ALIGN = 10.0
CONST_T_Y_ALIGN = 10.0
CONST_T_X_MULTIPLIER = 10.0
CONST_T_Y_MULTIPLIER = 10.0

# const n
CONST_N = 150

# eps impact
CONST_EPS_QUANTITY = CONST_N * 2
CONST_EPS = 1e-2

CONST_THRESHOLD = 1e-10

# const detla_t
CONST_DELTA = 1e-2

times = [i * CONST_DELTA for i in xrange(CONST_EPS_QUANTITY)]
times_2 = times[0:150]

# const excel file name
CONST_EXCEL_FILE_NAME = "gubaref.xlsx"

# create workbook
wb = Workbook()
# get active worksheet
ws = wb.active

series11 = [random.random() for i in xrange(CONST_SIZE)]
series12 = [random.random() for i in xrange(CONST_SIZE)]

series21 = [random.random() for i in xrange(CONST_SIZE)]
series22 = [random.random() for i in xrange(CONST_SIZE)]

series31 = [random.random() for i in xrange(CONST_SIZE)]
series32 = [random.random() for i in xrange(CONST_SIZE)]

series1 = []
series2 = []
series3 = []

for i in xrange(CONST_SIZE):
    # first
    series11[i] = series11[i] * CONST_FIRST_MULTIPLIER + CONST_FIRST_ALIGN
    series12[i] /= CONST_DIVISOR
    series1.append(complex(series12[i], series11[i]))
    # second
    series21[i] = series21[i] * CONST_SECOND_MULTIPLIER + CONST_SECOND_ALIGN
    series22[i] = series21[i] / CONST_DIVISOR
    series2.append(complex(series21[i], series22[i]))
    # third
    series31[i] = (series31[i] - 0.5) * CONST_T_X_MULTIPLIER + CONST_T_X_ALIGN
    series32[i] = (series32[i] - 0.5) * CONST_T_Y_MULTIPLIER + CONST_T_Y_ALIGN
    series3.append(complex(series31[i], series32[i]))

series = series1 + series2 + series3

# f and g funcs o_O
fs = [(random.random() * 0.3 + 0.7) * (random.randint(0, 1) * 2 - 1) for i in xrange(3 * CONST_SIZE)]
fc = [(random.random() * 0.3 + 0.7) * (random.randint(0, 1) * 2 - 1) for i in xrange(3 * CONST_SIZE)]

# define h,y,h_back
h = [0.0] * CONST_EPS_QUANTITY
y = [0.0] * CONST_EPS_QUANTITY
h_back = [0.0] * CONST_EPS_QUANTITY

# choosing eps
eps = [2.0 * CONST_EPS * (random.random() - 0.5) for i in xrange(CONST_EPS_QUANTITY)]


# <print>
ws.append(["our random values"])
ws.append(["Real"] + map(lambda x: x.real, series))
ws.append(["Imaginary"] + map(lambda x: x.imag, series))
ws.append([])
ws.append(["function"])
ws.append(["fs"] + fs)
ws.append(["fc"] + fc)
ws.append([])
ws.append(["epsilon impact"])
ws.append(range(1, CONST_EPS_QUANTITY + 1))
ws.append(eps)
# <print>


# h calc from eps
for j in xrange(1, CONST_EPS_QUANTITY + 1):
    sum = 0.0
    for p in xrange(CONST_FUNC_SIZE):
        sum += (fc[p] * math.cos(series[p].imag * CONST_DELTA * j) + fs[p] * math.sin(series[p].imag * CONST_DELTA * j)) * math.exp(-series[p].real * CONST_DELTA * j)
    h[j - 1] = sum

# y calc
for k in xrange(CONST_EPS_QUANTITY):
    sum = 0.0
    for j in xrange(k + 1):
        sum += h[j] * eps[k - j]
    y[k] = sum

# h calc from y eps
for i in xrange(0, CONST_EPS_QUANTITY):
    h_back[i] = y[i]
    for j in xrange(0, i):
        h_back[i] -= h[j] * eps[i - j]
    h_back[i] /= eps[0]


# <print>
ws.append([])
ws.append(["h0j"])
ws.append(range(1, CONST_EPS_QUANTITY + 1))
ws.append(h)
ws.append([])
ws.append(["y"])
ws.append(range(1, CONST_EPS_QUANTITY + 1))
ws.append(y)
ws.append([])
ws.append(["h_back"])
ws.append(range(1, CONST_EPS_QUANTITY + 1))
ws.append(h_back)
# <print>


error = 0.0
for i in xrange(0, CONST_EPS_QUANTITY):
    error += abs(h[i] - h_back[i])
print "error -", error

H_list = [h[i:i + CONST_N] for i in xrange(CONST_N)]
H = numpy.matrix(H_list)

S, V, D = numpy.linalg.svd(H, True)

model_size = 0
while V[model_size] > CONST_THRESHOLD:
    model_size += 1


if CONST_WRITE_SVD:
    # <print>
    ws.append([])
    ws.append(["H matrix"])
    for i in H.getA():
        ws.append(i.tolist())
    ws.append([])
    ws.append(["S matrix"])
    for i in S.getA():
        ws.append(i.tolist())
    ws.append([])
    ws.append(["Singular values"])
    ws.append(V.tolist())
    ws.append([])
    ws.append(["D matrix"])
    for i in D.getA():
        ws.append(i.tolist())
    ws.append([])
    ws.append([])
    ws.append(["system dim is like", model_size])
    # <print>


S_cut = S[range(model_size)][:, range(model_size)]
V_cut = numpy.diag(V[0:model_size])
D_cut = D[range(model_size)][:, range(model_size)]
H_cut = numpy.dot(numpy.dot(S_cut, V_cut), D_cut)
eigenvalues, eigenvectors = numpy.linalg.eig(H_cut)


# <print>
ws.append([])
ws.append(["S matrix (cut)"])
for i in S_cut.getA():
    ws.append(i.tolist())
ws.append([])
ws.append(["V matrix (cut)"])
for i in V_cut:
    ws.append(i.tolist())
ws.append([])
ws.append(["D matrix (cut)"])
for i in D_cut.getA():
    ws.append(i.tolist())
ws.append([])
ws.append([])
ws.append([])
ws.append(["H matrix (cut)"])
for i in H_cut.getA():
    ws.append(i.tolist())
ws.append([])
ws.append(["eigenvalues of H_cut"])
ws.append(["Real"] + map(lambda x: x.real, eigenvalues))
ws.append(["Imaginary"] + map(lambda x: x.imag, eigenvalues))

# </print>


# showing all data via matplotlib
if CONST_SHOW_PLOTS:
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.axis([-1, 15, -1, 15])
    plt.plot([series[i].real for i in xrange(len(series))], [series[i].imag for i in xrange(len(series))], "mo")
    plt.show()

    fig, axes = plt.subplots(2, 1, True)
    axes[0].plot(fc)
    axes[1].plot(fs)
    plt.show()

    plt.plot(times, eps)
    plt.show()

    fig, axes = plt.subplots(3, 1, True)
    axes[0].plot(times, h)
    axes[1].plot(times, y)
    axes[2].plot(times, h_back)
    plt.show()

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.plot([eigenvalues[i].real for i in xrange(len(eigenvalues))],
             [eigenvalues[i].imag for i in xrange(len(eigenvalues))], "mo")
    plt.show()

    Srow1 = S[1,:].tolist()[0]
    Srow20 = S[20,:].tolist()[0]
    Dcol1 = D[:,1]
    Dcol20 = D[:,20]

    fig, axes = plt.subplots(4, 1, True)
    axes[0].plot(times_2, Srow1)
    axes[1].plot(times_2, Dcol1)
    axes[2].plot(times_2, Srow20)
    axes[3].plot(times_2, Dcol20)
    plt.show()

wb.save(CONST_EXCEL_FILE_NAME)
