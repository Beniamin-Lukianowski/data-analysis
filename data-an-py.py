import numpy as np 
import math 
import csv 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
  return [dlugosc_fali, np.array(intensywnosc)]

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

def przeliczanie_kata(piksel, binning, NA):
   return np.arctan(piksel*binning*np.tan(np.arcsin(NA))/240)*180/np.pi

def przeliczanie_kata_na_piksel(kat, binning, NA):
    return np.tan(kat*np.pi/180)/binning/np.tan(np.arcsin(NA))*240

def usuwanie_kosmikow(intensywnosc, liczba):
    lista_max = []
    for i in range (liczba):
        lista_max.append(0)
    lista_pos = np.zeros((liczba, 2))
    for i in range (len(intensywnosc)):
        for j in range (len(intensywnosc[i])):
            min_list = np.min(lista_max)
            pos_min_list = lista_max.index(min_list)
            if intensywnosc[i][j] > min_list:
                lista_max[pos_min_list] = intensywnosc[i][j]
                lista_pos[pos_min_list] = [i, j]
    if (np.max(lista_max) - np.min(lista_max))/np.min(lista_max) < 0.5:
        return intensywnosc/np.max(lista_max)
    else: 
        for i in range(liczba):
            if lista_max[i] > np.min(lista_max)*1.5:
                intensywnosc[int(lista_pos[i][0])][int(lista_pos[i][1])] = 0
        return intensywnosc/np.max(intensywnosc)

def pojedyncza_mapa(intensywnosc, dlugosc_fali, wektor_Stokesa, srodek, zakres, binning, NA):
    kat = przeliczanie_kata(np.arange(-zakres, zakres + 1, 1), binning, NA)
    kat_mesh, dlugosc_fali_mesh = np.meshgrid(kat, dlugosc_fali)
    color = ['inferno', 'PiYG', 'PuOr', 'seismic']
    int_max = np.max(intensywnosc)
    int_min = np.min(intensywnosc)
    int_min_max = max([abs(int_min), int_max])
    if int_min_max < 0.1 and wektor_Stokesa != 0:
        int_min_max = 0.1 
    elif wektor_Stokesa != 0:
        int_min_max = math.ceil(int_min_max*10)/10
    else:
        intensywnosc = 1 - usuwanie_kosmikow(intensywnosc, 5) 
    if wektor_Stokesa != 0:     
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))
        ax1.pcolor(kat_mesh, dlugosc_fali_mesh, intensywnosc[:, srodek - zakres : srodek + zakres + 1], cmap = color[wektor_Stokesa], vmax = int_min_max, vmin = -int_min_max)
        fig.colorbar(ax1.pcolor(kat_mesh, dlugosc_fali_mesh, intensywnosc[:, srodek - zakres : srodek + zakres + 1], cmap = color[wektor_Stokesa], vmax = int_min_max, vmin = -int_min_max), cax = ax1.inset_axes([0.1, 0.1, 0.05, 0.4]), ticks = [-int_min_max, 0, int_min_max])
    else: 
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))
        ax1.pcolor(kat_mesh, dlugosc_fali_mesh, intensywnosc[:, srodek - zakres : srodek + zakres + 1], cmap = color[wektor_Stokesa], vmax = 1, vmin = 0)
        fig.colorbar(ax1.pcolor(kat_mesh, dlugosc_fali_mesh, intensywnosc[:, srodek - zakres : srodek + zakres + 1], cmap = color[wektor_Stokesa], vmax = 1, vmin = 0), cax = ax1.inset_axes([0.1, 0.1, 0.05, 0.4]), ticks = [0, 0.5, 1])
    ax1.invert_yaxis()
    ax1.set_ylabel("$ \lambda \mathrm{\, [nm]} $")
    ax1.set_xlabel("$ \mathrm{Angle}\, [^\circ]$")
    ax1.tick_params(direction =  "in", bottom = True, top = True, left = True, right = True, labelbottom = True, labelleft = True, labelright = False, labeltop = False)

def funkcja_Lorentza(x, Amp, x0, gamma, ground):
    wartosc_funkcji = np.zeros(len(x))
    for i in range(len(x)):
        wartosc_funkcji[i] = Amp/(1 + ((x[i] - x0)/gamma)**2) + ground
    return wartosc_funkcji

def funkcja_Gaussa(x, Amp, x0, sigma, ground):
    wartosc_funkcji = np.zeros(len(x))
    for i in range(len(x)):
        wartosc_funkcji[i] = Amp*np.exp(-(x[i] - x0)**2/(2*sigma**2)) + ground
    return wartosc_funkcji

def dopasowanie_funkcji(funkcja, dlugosc_fali, intensywnosc, lambda_poczatkowe, lambda_koncowe, poczatkowe_parametry, wykres):
    krok = (dlugosc_fali[-1] - dlugosc_fali[0])/len(dlugosc_fali)
    poczatek = int((lambda_poczatkowe - data_H[0][0])/krok)
    koniec = int((lambda_koncowe - data_H[0][0])/krok)
    dane_x = dlugosc_fali[poczatek : koniec]
    dane_y = intensywnosc[poczatek : koniec]
    parametry, macierz_kowariancji = curve_fit(funkcja, dane_x, dane_y, p0 = poczatkowe_parametry, bounds = ([0, lambda_poczatkowe, 0, 0], [1e4, lambda_koncowe, 5, 100]))
    if wykres == 'tak':
      plt.plot(dane_x, dane_y)
      plt.plot(dane_x, funkcja(dane_x, parametry[0], parametry[1], parametry[2], parametry[3]))
      plt.show()
    return parametry 

def dopasowanie_widmo(funkcja, dlugosc_fali, intensywnosc, lambda_poczatkowe, lambda_koncowe, poczatkowe_parametry, srodek_kata, kat_koncowy, NA):
    delta_lambda = 5
    liczba_pikseli = int(przeliczanie_kata_na_piksel(kat_koncowy, 2048/len(dlugosc_fali), NA))
    lista_parametrow = np.zeros((2*liczba_pikseli + 1, 4))
    lista_parametrow[liczba_pikseli] = dopasowanie_funkcji(funkcja, dlugosc_fali, intensywnosc[:, srodek_kata], lambda_poczatkowe, lambda_koncowe, poczatkowe_parametry, 'nie')
    for i in range(1, liczba_pikseli + 1):
        lista_parametrow[liczba_pikseli - i] = dopasowanie_funkcji(funkcja, dlugosc_fali, intensywnosc[:, srodek_kata - i], lista_parametrow[liczba_pikseli - i + 1][1] - delta_lambda, lista_parametrow[liczba_pikseli - i + 1][1] + delta_lambda, lista_parametrow[liczba_pikseli - i + 1], 'nie')
        lista_parametrow[liczba_pikseli + i] = dopasowanie_funkcji(funkcja, dlugosc_fali, intensywnosc[:, srodek_kata + i], lista_parametrow[liczba_pikseli + i - 1][1] - delta_lambda, lista_parametrow[liczba_pikseli + i - 1][1] + delta_lambda, lista_parametrow[liczba_pikseli + i - 1], 'nie')
    return lista_parametrow

przeliczanie_kata_na_piksel(2, 2, 0.6)

dzien = '2022_07_26_'
sciezka_podstawa = 'Data/image_'
sciezka = sciezka_podstawa + dzien
data_H = wczytywanie_pliku(sciezka, 1)
data_V = wczytywanie_pliku(sciezka, 2)

parametry = dopasowanie_widmo(funkcja_Lorentza, data_H[0], intensywnosc_bez_tla(data_H[1], 'lin'), 585, 600, [500, 590, 0.4, 15], 130, 30, 0.6)

plt.plot(przeliczanie_kata(np.arange(-(len(parametry) - 1)/2, (len(parametry) - 1)/2 + 1, 1), 2, 0.6), parametry[:, 1])

print(len(np.arange(-(len(parametry) - 1)/2, (len(parametry) - 1)/2 + 1, 1)))
print(len(parametry))

plt.savefig('sciezka', dpi = 200)

pojedyncza_mapa(polaryzacja(intensywnosc_bez_tla(data_H[1], 'lin'), intensywnosc_bez_tla(data_V[1], 'lin')), data_H[0], 1, 130, 80, 2, 0.6)

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