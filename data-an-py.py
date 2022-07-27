
import numpy as np 
import math 
import csv 
import matplotlib.pyplot as plt


def wczytywanie_pliku(sciezka, numer):
  dlugosc_fali = []
  intensywnosc = []
  with open(sciezka + str(numer).rjust(3, '0') + '.asc', 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter  = '\t')
      for row in reader:
          intensywnosc_pom = []
          for i in range(1, len(row)-1):
              intensywnosc_pom.append(float(row[i]))
          intensywnosc.append(intensywnosc_pom)
          dlugosc_fali.append(float(row[0]))
  return [dlugosc_fali, intensywnosc]

def intensywnosc_bez_tla(intensywnosc, skala):
  intensywnosc_min = np.min(intensywnosc) - 2
  for i in range(len(intensywnosc)):
    for j in range(len(intensywnosc[i])):
      if skala == 'lin':
        intensywnosc[i][j] = intensywnosc[i][j] - intensywnosc_min
      elif skala == 'log':
        intensywnosc[i][j] = np.log10(intensywnosc[i][j] - intensywnosc_min)
      else: 
        print('Nie ma takiej skali')
  return intensywnosc

def polaryzacja(int_1, int_2):
  polaryzacja = np.zeros((len(int_1), len(int_1[0])))
  for i in range (len(int_1)):
    for j in range (len(int_1[i])):
      polaryzacja[i][j] = (int_1[i][j] - int_2[i][j])/(int_1[i][j] + int_2[i][j])
  return polaryzacja

dzien = '2022_07_26_'
sciezka_podstawa = '/Data/image_'
sciezka = sciezka_podstawa + dzien
data_H = wczytywanie_pliku(sciezka, 1)
data_V = wczytywanie_pliku(sciezka, 2)

kat = np.linspace(1, 256, 256)
kat_mesh, dlugosc_fali_mesh = np.meshgrid(kat, data_H[0])
S1 = polaryzacja(intensywnosc_bez_tla(data_H[1], 'lin'), intensywnosc_bez_tla(data_V[1], 'lin'))
max_S1 = np.max(S1)
min_S1 = np.min(S1)
min_max_S1 = max([abs(min_S1), max_S1])
if min_max_S1 < 0.1:
  min_max_S1 = 0.1 
else:
  min_max_S1 = math.ceil(min_max_S1*10)/10
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
fig.subplots_adjust(hspace = 0, wspace = 0, top = 0.92, bottom = 0.08, right = 0.96, left = 0.08)
ax1.pcolor(kat_mesh, dlugosc_fali_mesh, 1 - intensywnosc_bez_tla(data_H[1], 'lin')/np.max(intensywnosc_bez_tla(data_H[1], 'lin')), cmap = 'inferno', vmin = 0, vmax = 1)
ax1.invert_yaxis()
fig.colorbar(ax1.pcolor(kat_mesh, dlugosc_fali_mesh,  1 - intensywnosc_bez_tla(data_H[1], 'lin')/np.max(intensywnosc_bez_tla(data_H[1], 'lin')), cmap = 'inferno', vmax = 1, vmin = 0), cax = ax1.inset_axes([0.1, 0.1, 0.05, 0.4]), ticks = [0, 0.5, 1])
ax1.tick_params(direction =  "in", bottom = True, top = True, left = True, right = True, labelbottom = True, labelleft = True, labelright = False, labeltop = False, color = "black")
ax2.pcolor(kat_mesh, dlugosc_fali_mesh, 1 - intensywnosc_bez_tla(data_V[1], 'lin')/np.max(intensywnosc_bez_tla(data_V[1], 'lin')), cmap = 'inferno', vmin = 0, vmax = 1)
ax2.invert_yaxis()
fig.colorbar(ax2.pcolor(kat_mesh, dlugosc_fali_mesh,  1 - intensywnosc_bez_tla(data_V[1], 'lin')/np.max(intensywnosc_bez_tla(data_V[1], 'lin')), cmap = 'inferno', vmax = 1, vmin = 0), cax = ax2.inset_axes([0.1, 0.1, 0.05, 0.4]), ticks = [0, 0.5, 1])
ax2.tick_params(direction =  "in", bottom = True, top = True, left = True, right = True, labelbottom = True, labelleft = False, labelright = False, labeltop = False, color = "black")
ax3.pcolor(kat_mesh, dlugosc_fali_mesh, S1, cmap = 'PiYG', vmax = min_max_S1, vmin = -min_max_S1)
ax3.invert_yaxis()
fig.colorbar(ax3.pcolor(kat_mesh, dlugosc_fali_mesh, S1, cmap = 'PiYG', vmax = min_max_S1, vmin = -min_max_S1), cax = ax3.inset_axes([0.1, 0.1, 0.05, 0.4]), ticks = [-min_max_S1, 0, min_max_S1])
ax3.tick_params(direction =  "in", bottom = True, top = True, left = True, right = True, labelbottom = True, labelleft = False, labelright = False, labeltop = False, color = "black")

print(np.max(intensywnosc_bez_tla(data_H[1], 'lin')))
print(np.min(intensywnosc_bez_tla(data_H[1], 'lin')))

kat = np.linspace(1, 230, 230)
kat_mesh, dlugosc_fali_mesh = np.meshgrid(kat, data_H[0])

fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
ax1.pcolor(kat_mesh, dlugosc_fali_mesh, intensywnosc_bez_tla(data_H[1], 'lin'))
ax1.invert_yaxis()

fig, (ax1) = plt.subplots(1, 1, figsize=(3, 4))
ax1.pcolor(kat_mesh, dlugosc_fali_mesh, data_H[1])
ax1.invert_yaxis()
plt.show()
fig, (ax1) = plt.subplots(1, 1, figsize=(3, 4))
ax1.pcolor(kat_mesh, dlugosc_fali_mesh, data_V[1])
ax1.invert_yaxis()
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4))
ax1.pcolor(kat_mesh, dlugosc_fali_mesh, intensywnosc_bez_tla(data_H[1], 'lin'))
ax1.invert_yaxis()
ax2.pcolor(kat_mesh, dlugosc_fali_mesh, intensywnosc_bez_tla(data_V[1], 'lin'))
ax2.invert_yaxis()
ax3.pcolor(kat_mesh, dlugosc_fali_mesh, polaryzacja(intensywnosc_bez_tla(data_H[1], 'lin'), intensywnosc_bez_tla(data_V[1], 'lin')))
ax3.invert_yaxis()
plt.show()

plt.imshow(data_H[1])
plt.show()
fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))
ax1.imshow(data_H[1])
plt.show()
