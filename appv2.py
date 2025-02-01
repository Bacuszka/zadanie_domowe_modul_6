import pandas as pd
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline
import seaborn as sns
import os
import json





#
# MAIN PROGRAM
#
df=pd.read_csv("35__welcome_survey_cleaned.csv", sep=";")

st.title("\U0001F4DA Ankieta powitalna \U0000270F")

#ZMIANA NAZWY

new_column_names = {
    "age": "Przedział wiekowy",
    "edu_level": "Wykształcenie",
    "fav_animals": "Ulubione zwierzę",
    "fav_place": "Ulubione miejsce",
    "gender": "Płeć",
    "hobby_art": "Hobby sztuka",
    "hobby_books": "Hobby książki",
    "hobby_movies": "Hobby filmy",
    "hobby_other": "Hobby inne",
    "hobby_sport": "Hobby sport",
    "hobby_video_games": "Hobby gry wideo",
    "industry": "Branża",
    "learning_pref_books": "Preferowany sposób nauki z książki",
    "learning_pref_chatgpt": "Preferowany sposób nauki ChatGPT",
    "learning_pref_offline_courses": "Preferowany sposób nauki z kursów offline",
    "learning_pref_online_courses": "Preferowany sposób nauki z kursów online",
    "learning_pref_personal_projects": "Preferowany sposób nauki z projektów osobistych",
    "learning_pref_teaching": "Preferowany sposób nauki z nauczycielem",
    "learning_pref_teamwork": "Preferowany sposób nauki pracy zespołowej",
    "learning_pref_workshops": "Preferowany sposób nauki warsztaty",
    "motivation_career": "Motywacja kariera",
    "motivation_challenges": "Motywacja wyzwania",
    "motivation_creativity_and_innovation": "Motywacja kreatywność i innowacje",
    "motivation_money_and_job": "Motywacja pieniądze i praca",
    "motivation_personal_growth": "Motywacja rozwój osobisty",
    "motivation_remote": "Motywacja praca zdalna",
    "sweet_or_salty": "Słony czy słodki",
    "years_of_experience": "lata doświadczenia"
}

# Zmiana nazw kolumn
df.rename(columns=new_column_names, inplace=True)
df['Płeć'] = df['Płeć'].map({1.0: 'Mężczyzna', 0.0: 'Kobieta'})

#
#ZMIANA DO WYKRESÓW
#

opcje = {
    "Przedział wiekowy": 'Przedział wiekowy',
    "Wykształcenie": 'Wykształcenie',
    "Ulubione zwierzę": 'Ulubione zwierzę',
    "Ulubione miejsce": 'Ulubione miejsce',
    "Płeć": 'Płeć',
    "Hobby": ["Hobby sztuka", "Hobby książki", "Hobby filmy", "Hobby sport", "Hobby gry wideo", "Hobby inne"],
    "Branża": 'Branża',
    "Preferowany sposób nauki": [
        "Preferowany sposób nauki z książki",
        "Preferowany sposób nauki ChatGPT",
        "Preferowany sposób nauki z kursów offline",
        "Preferowany sposób nauki z kursów online"
    ],
    "Motywacja": [
        "Motywacja kariera", "Motywacja wyzwania"
    ],
    "Słony czy słodki": 'Słony czy słodki',
    "lata doświadczenia": 'lata doświadczenia'
}

# Funkcja do tworzenia gradientu koloru
def get_colors(values, cmap_name="Greens"):
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(min(values), max(values))
    return [cmap(norm(value)) for value in values]

def rysuj_wykres(kolumna):
    if isinstance(kolumna, list):  # Suma dla grup kolumn
        counts = df[kolumna].sum()
    else:  # Zliczanie wartości w jednej kolumnie
        counts = df[kolumna].value_counts(dropna=False)
    
    # Tworzenie gradientu kolorów
    kolory = get_colors(counts.values)
    
    # Wykres
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax, color=kolory, edgecolor="black")
    
    # Dodanie tytułu i etykiet
    ax.set_title(f"Wykres słupkowy dla: {kolumna}", fontsize=16)
    ax.set_xlabel("Kategorie", fontsize=12)
    ax.set_ylabel("Liczba osób", fontsize=12)
    
    # Dodanie siatki
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ustawienie odstępów osi Y
    max_value = max(counts.values)
    ax.set_yticks(range(0, max_value + 1, 4))
    
    # Wyświetlenie wykresu
    st.pyplot(fig)
#
# SIDE BAR
#
# Boczne menu
with st.sidebar:
    st.subheader("Wybierz opcje:")

#
# NOTATNIK W SIDEBARZE
#


#Wybierz wykres słupkowy
wybor = st.sidebar.selectbox("Wybierz wykres słupkowy", list(opcje.keys()))


# Sidebar - Notatnik
with st.sidebar:
    st.header("Notatnik")

    # Pole do wpisywania notatki
    notatka = st.text_area(
        "Wpisz notatkę",
        height=200,
        max_chars=1000,
    )

#
# MAIN PROGRAM
#

liczba_mezczyzn = df[df["Płeć"] == 1].shape[0]  # Filtrujemy wiersze, gdzie Płeć == 1 i liczymy wiersze

# Liczba kobiet
liczba_kobiet = df[df["Płeć"] == 0].shape[0]  # Filtrujemy wiersze, gdzie Płeć == 0 i liczymy wiersze

# Liczba osób bez podanej płci (NaN lub inne wartości)
liczba_brak_podanej_pleci = df[df["Płeć"].isna()].shape[0]  # Filtrujemy wiersze, gdzie Płeć jest NaN

# Można też użyć value_counts, aby obliczyć liczbę każdej wartości, w tym NaN
statystyki_plec = df["Płeć"].value_counts(dropna=False)
liczba_mezczyzn = statystyki_plec.get(1, 0)  # Wartość 1 oznacza mężczyzn
liczba_kobiet = statystyki_plec.get(0, 0)  # Wartość 0 oznacza kobiety
liczba_brak_podanej_pleci = statystyki_plec.get(np.nan, 0)  # Wartość NaN oznacza brak podanej płci

# Wyświetlenie wyników w Streamlit
c0, c1, c2, c3 = st.columns(4)  # Dodajemy c3 dla kategorii "Brak podanej płci"
with c0:
    st.metric("Liczba uczestników", df.shape[0])  # Liczba wszystkich uczestników
with c1:
    st.metric("Liczba mężczyzn", liczba_mezczyzn)
with c2:
    st.metric("Liczba kobiet", liczba_kobiet)
with c3:
    st.metric("Brak podanej płci / inna", liczba_brak_podanej_pleci)

    
# 10 losowych wierszy
x = min(10, len(df))
st.write(f"## {x} losowych wierszy")
st.dataframe(df.sample(x), use_container_width=True, hide_index=True)


# Wyświetlanie wykresu
if wybor:
    kolumna = opcje[wybor]
    rysuj_wykres(kolumna)


#
# HISTOGRAM
#


# Funkcja do rysowania histogramu z linią trendu
def rysuj_histogram_z_trendem_posortowany():
    # Zmapuj kategorie na wartości liczbowe
    przedzial_map = {
        "<18": 9,  # Środek przedziału
        "18-24": 21,
        "25-34": 29.5,
        "35-44": 39.5,
        "45-54": 49.5,
        "55-64": 59.5,
        ">=65": 70,
        "unknown": np.nan  # NaN dla unknown
    }
    
    # Zastąp wartości w kolumnie "Przedział wiekowy" odpowiednimi liczbami
    df["Przedział wiekowy numerycznie"] = df["Przedział wiekowy"].map(przedzial_map)
    
    # Oblicz średnią z wartości numerycznych, ignorując NaN
    srednia = df["Przedział wiekowy numerycznie"].mean()
    
    # Zamień "unknown" na średnią
    df["Przedział wiekowy numerycznie"].fillna(srednia, inplace=True)
    
    # Przywróć oryginalne kategorie
    odwrotny_map = {v: k for k, v in przedzial_map.items()}
    odwrotny_map[srednia] = "Średnia (z unknown)"
    df["Przedział wiekowy posortowany"] = df["Przedział wiekowy numerycznie"].map(odwrotny_map)
    
    # Określ niestandardowy porządek kategorii
    porzadek_kategorii = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65", "Średnia (z unknown)"]
    df["Przedział wiekowy posortowany"] = pd.Categorical(
        df["Przedział wiekowy posortowany"],
        categories=porzadek_kategorii,
        ordered=True
    )
    
    # Zliczenie wartości
    counts = df["Przedział wiekowy posortowany"].value_counts().sort_index()

    # Numerowanie kategorii
    x_numeric = np.arange(len(counts))
    liczby = counts.values

    # Tworzenie linii trendu
    spline = make_interp_spline(x_numeric, liczby, k=3)
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 300)
    y_smooth = spline(x_smooth)

    # Rysowanie wykresu
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Histogram
    ax.bar(x_numeric, liczby, color="#36d736", edgecolor="black", label="Liczba osób")
    
    # Linia trendu
    ax.plot(x_smooth, y_smooth, color="red", linewidth=2, label="Linia trendu")

    # Etykiety osi i tytuł
    ax.set_xticks(x_numeric)
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=10)
    ax.set_title("Histogram przedziałów wiekowych (posortowany) z linią trendu", fontsize=16)
    ax.set_xlabel("Przedział wiekowy", fontsize=12)
    ax.set_ylabel("Liczba osób", fontsize=12)
    
    # Dodanie siatki
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Dodanie legendy
    ax.legend(fontsize=12)

    # Wyświetlenie wykresu w Streamlit
    st.pyplot(fig)

# Wywołanie funkcji
rysuj_histogram_z_trendem_posortowany()

#
# WYKRES KOŁOWY
#

# Funkcja do rysowania wykresu kołowego dla Hobby
def rysuj_wykres_kolowy_hobby():
    # Zdefiniuj kolumny dla hobby
    kolumny_hobby = [
        "Hobby sztuka", "Hobby książki", "Hobby filmy", 
        "Hobby sport", "Hobby gry wideo", "Hobby inne"
    ]
    
    # Zsumuj wartości dla każdego hobby
    suma_hobby = df[kolumny_hobby].sum()
    
    # Oblicz procenty
    procenty = (suma_hobby / suma_hobby.sum()) * 100
    
    # Tworzenie nazw dla etykiet bez słowa "Hobby"
    etykiety = ["Sztuka", "Książki", "Filmy", "Sport", "Gry Wideo", "Inne"]
    
    # Kolory zielonkawe
    kolory = ['#d9f2d9', '#adebad', '#80d480', '#54c054', '#2bab2b', '#008f00']
    
    # Tworzenie wykresu kołowego
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        suma_hobby,
        labels=etykiety,
        autopct='%1.1f%%',
        colors=kolory,
        startangle=90,
        textprops=dict(color="black")
    )
    
    # Ustawienia wykresu
    ax.set_title("Udział różnych hobby w ankiecie", fontsize=16)
    plt.setp(autotexts, size=10, weight="bold")
    
    # Wyświetlenie wykresu w Streamlit
    st.pyplot(fig)

# Wywołanie funkcji
rysuj_wykres_kolowy_hobby()

#
# SKRZYPCE
#

experience_order = ['0-2', '3-5', '6-10', '11-15', '>=16']

# Tworzenie wykresu skrzypcowego z ustaloną kolejnością
plt.figure(figsize=(10, 6))
sns.violinplot(
    x="lata doświadczenia",
    y="Ulubione miejsce",
    data=df,
    palette="Greens",
    order=experience_order  # Ustawienie kolejności osi X
)

# Dodanie etykiet i tytułu
plt.title("Wykres skrzypcowy: Lata doświadczenia vs. Ulubione miejsce", fontsize=14)
plt.xlabel("Lata doświadczenia", fontsize=12)
plt.ylabel("Ulubione miejsce", fontsize=12)

# Wyświetlenie wykresu
st.pyplot(plt.gcf())


#
# HEATMAPA
#


st.write("## Macierz korelacji")
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Greens', cbar=True)
st.pyplot(plt.gcf())


#
# SCATTRPLOT
#

