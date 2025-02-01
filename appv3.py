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
import plotly.express as px

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
    st.title("OPCJE:")
    st.header("- Wybierz filtry:")


#
# WYBORY FILTRY
#

    filtered_df = df.copy()


# Lista rozwijana dla "Płeć"
gender_options = ["Mężczyzna", "Kobieta", "Brak/Inne"]
selected_gender = st.sidebar.selectbox(
    "Wybierz płeć",
    options=["Wszystkie"] + gender_options,
)

# Filtrowanie danych po wybranej płci
if selected_gender != "Wszystkie":
    if selected_gender == "Kobieta":
        filtered_df = filtered_df[filtered_df["Płeć"] == "Kobieta"]
    elif selected_gender == "Mężczyzna":
        filtered_df = filtered_df[filtered_df["Płeć"] == "Mężczyzna"]
    else:  # Dla "Brak/Inne"
        filtered_df = filtered_df[filtered_df["Płeć"].isna() | (filtered_df["Płeć"] == "Inne")]


# Lista rozwijana dla "Przedział wiekowy"
age_options = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']
selected_age = st.sidebar.selectbox(
    "Wybierz przedział wiekowy",
    options=["Wszystkie"] + age_options,
)

# Filtrowanie danych po wybranym przedziale wiekowym
if selected_age != "Wszystkie":
    if selected_age == "<18":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == "<18"]
    elif selected_age == "18-24":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == "18-24"]
    elif selected_age == "25-34":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == "25-34"]
    elif selected_age == "35-44":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == "35-44"]
    elif selected_age == "45-54":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == "45-54"]
    elif selected_age == "55-64":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == "55-64"]
    elif selected_age == ">=65":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == ">=65"]
    elif selected_age == "unknown":
        filtered_df = filtered_df[filtered_df["Przedział wiekowy"] == "unknown"]

#ZWIERZĘTA
# Lista rozwijana dla "Ulubione zwierzę"
animal_options = ["Psy", "Koty", "Inne", "Koty i Psy", "Brak ulubionych"]
selected_animal = st.sidebar.selectbox(
    "Wybierz ulubione zwierzę",
    options=["Wszystkie"] + animal_options,
)
#
# Filtrowanie danych po wybranym zwierzęciu
#
if selected_animal != "Wszystkie":
    if selected_animal == "Psy":
        filtered_df = filtered_df[filtered_df["Ulubione zwierzę"] == "Psy"]
    elif selected_animal == "Koty":
        filtered_df = filtered_df[filtered_df["Ulubione zwierzę"] == "Koty"]
    elif selected_animal == "Koty i Psy":
        filtered_df = filtered_df[filtered_df["Ulubione zwierzę"] == "Koty i Psy"]
    elif selected_animal == "Brak ulubionych":
        filtered_df = filtered_df[filtered_df["Ulubione zwierzę"] == "Brak ulubionych"]
    else:  # Dla "Inne"
        filtered_df = filtered_df[filtered_df["Ulubione zwierzę"] == "Inne"]
#
# ULUBIONE MIEJSCE
#
place_options = ["Nad wodą", "W lesie", "W górach", "Inne"]
selected_place = st.sidebar.selectbox(
    "Wybierz ulubione miejsce",
    options=["Wszystkie"] + place_options,
)

# Filtrowanie danych po wybranym miejscu
if selected_place != "Wszystkie":
    if selected_place == "Nad wodą":
        filtered_df = filtered_df[filtered_df["Ulubione miejsce"] == "Nad wodą"]
    elif selected_place == "W lesie":
        filtered_df = filtered_df[filtered_df["Ulubione miejsce"] == "W lesie"]
    elif selected_place == "W górach":
        filtered_df = filtered_df[filtered_df["Ulubione miejsce"] == "W górach"]
    elif selected_place == "Inne":
        filtered_df = filtered_df[filtered_df["Ulubione miejsce"] == "Inne"]

# Lista rozwijana dla "Hobby"
hobby_options = ["Sztuka", "Książki", "Filmy", "Sport", "Gry wideo", "Inne"]
selected_hobby = st.sidebar.selectbox(
    "Wybierz hobby",
    options=["Wszystkie"] + hobby_options,
)

# Filtrowanie danych po wybranym hobby
if selected_hobby != "Wszystkie":
    if selected_hobby == "Sztuka":
        filtered_df = filtered_df[filtered_df["Hobby sztuka"] == 1]
    elif selected_hobby == "Książki":
        filtered_df = filtered_df[filtered_df["Hobby książki"] == 1]
    elif selected_hobby == "Filmy":
        filtered_df = filtered_df[filtered_df["Hobby filmy"] == 1]
    elif selected_hobby == "Sport":
        filtered_df = filtered_df[filtered_df["Hobby sport"] == 1]
    elif selected_hobby == "Gry wideo":
        filtered_df = filtered_df[filtered_df["Hobby gry wideo"] == 1]
    elif selected_hobby == "Inne":
        filtered_df = filtered_df[filtered_df["Hobby inne"] == 1]

# Lista rozwijana dla "Branża"
industry_options = [
    'IT', 'Edukacja', 'Energetyka', 'Automotive', 'Automatyzacja', 'Energetyka zawodowa',
    'Zdrowie', 'Kadry (HR)', 'Marketing', 'Produkcja', 'Wellness', 'Chemia', 'Nieruchomości',
    'Poligrafia', 'Administracja publiczna', 'Usługi', 'Obsługa klienta', 'Brak', 'Budowlana',
    'Automatyka i robotyka', 'Bezrobotny', 'Finanse', 'Inżynieria', 'Opieka', 'Emerytura',
    'Hotelarstwo', 'Logistyka', 'Motoryzacja', 'R&D', 'Ochrona Środowiska', 'Pomoc Społeczna',
    'Transport', 'Logistyka i Produkcja', 'Bezpieczeństwo', 'Logistyka i Transport', 'Budownictwo',
    'Architektura', 'E-commerce', 'Gastronomia'
]

selected_industry = st.sidebar.selectbox(
    "Wybierz branżę",
    options=["Wszystkie"] + industry_options,
)


# Lista rozwijana dla "Lata doświadczenia"
experience_options = ['0-2', '3-5', '6-10', '11-15', '>=16', 'unknown']
selected_experience = st.sidebar.selectbox(
    "Wybierz lata doświadczenia",
    options=["Wszystkie"] + experience_options,
)

# Filtrowanie danych po wybranych latach doświadczenia
if selected_experience != "Wszystkie":
    if selected_experience == "0-2":
        filtered_df = filtered_df[filtered_df["lata doświadczenia"] == "0-2"]
    elif selected_experience == "3-5":
        filtered_df = filtered_df[filtered_df["lata doświadczenia"] == "3-5"]
    elif selected_experience == "6-10":
        filtered_df = filtered_df[filtered_df["lata doświadczenia"] == "6-10"]
    elif selected_experience == "11-15":
        filtered_df = filtered_df[filtered_df["lata doświadczenia"] == "11-15"]
    elif selected_experience == ">=16":
        filtered_df = filtered_df[filtered_df["lata doświadczenia"] == ">=16"]
    elif selected_experience == "unknown":
        filtered_df = filtered_df[filtered_df["lata doświadczenia"].isna()]

# Filtrowanie danych po wybranej branży
if selected_industry != "Wszystkie":
    filtered_df = filtered_df[filtered_df["Branża"] == selected_industry]


# Lista rozwijana dla "Preferowany sposób nauki"
learning_options = [
    "Z książki", "ChatGPT", "Z kursów offline", "Z kursów online", "Z projektów osobistych",
    "Z nauczycielem", "Praca zespołowa", "Warsztaty"
]

selected_learning = st.sidebar.selectbox(
    "Wybierz preferowany sposób nauki",
    options=["Wszystkie"] + learning_options,
)

# Filtrowanie danych po wybranym preferowanym sposobie nauki
if selected_learning != "Wszystkie":
    if selected_learning == "Z książki":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki z książki"] == 1]
    elif selected_learning == "ChatGPT":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki ChatGPT"] == 1]
    elif selected_learning == "Z kursów offline":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki z kursów offline"] == 1]
    elif selected_learning == "Z kursów online":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki z kursów online"] == 1]
    elif selected_learning == "Z projektów osobistych":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki z projektów osobistych"] == 1]
    elif selected_learning == "Z nauczycielem":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki z nauczycielem"] == 1]
    elif selected_learning == "Praca zespołowa":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki pracy zespołowej"] == 1]
    elif selected_learning == "Warsztaty":
        filtered_df = filtered_df[filtered_df["Preferowany sposób nauki warsztaty"] == 1]

# Lista rozwijana dla "Motywacja"
motivation_options = [
    "Kariera", "Wyzwania", "Kreatywność i innowacje", "Pieniądze i praca",
    "Rozwój osobisty", "Praca zdalna"
]

selected_motivation = st.sidebar.selectbox(
    "Wybierz motywację",
    options=["Wszystkie"] + motivation_options,
)

# Filtrowanie danych po wybranej motywacji
if selected_motivation != "Wszystkie":
    if selected_motivation == "Kariera":
        filtered_df = filtered_df[filtered_df["Motywacja kariera"] == 1]
    elif selected_motivation == "Wyzwania":
        filtered_df = filtered_df[filtered_df["Motywacja wyzwania"] == 1]
    elif selected_motivation == "Kreatywność i innowacje":
        filtered_df = filtered_df[filtered_df["Motywacja kreatywność i innowacje"] == 1]
    elif selected_motivation == "Pieniądze i praca":
        filtered_df = filtered_df[filtered_df["Motywacja pieniądze i praca"] == 1]
    elif selected_motivation == "Rozwój osobisty":
        filtered_df = filtered_df[filtered_df["Motywacja rozwój osobisty"] == 1]
    elif selected_motivation == "Praca zdalna":
        filtered_df = filtered_df[filtered_df["Motywacja praca zdalna"] == 1]


# Lista rozwijana dla "Słony czy słodki"
taste_options = ["Słodki", "Słony"]
selected_taste = st.sidebar.selectbox(
    "Wybierz smak",
    options=["Wszystkie"] + taste_options,
)

# Filtrowanie danych po wybranym smaku
if selected_taste != "Wszystkie":
    if selected_taste == "Słodki":
        filtered_df = filtered_df[filtered_df["Słony czy słodki"] == "sweet"]
    elif selected_taste == "Słony":
        filtered_df = filtered_df[filtered_df["Słony czy słodki"] == "salty"]
#
# NOTATNIK W SIDEBARZE
#

st.sidebar.header("- Wykresy")
#Wybierz wykres słupkowy
wybor = st.sidebar.selectbox("Wybierz wykres słupkowy", list(opcje.keys()))


# Sidebar - Notatnik
with st.sidebar:
    st.header("- Notatnik")

    # Pole do wpisywania notatki
    notatka = st.text_area(
        "Wpisz notatkę. \nUWAGA! Po odświeżeniu strony notatki przepadną",
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

    
x = min(10, len(df))

# Stworzenie rozwijanego paska
with st.expander(f"Zobacz {x} losowych wierszy. Dane w tej tabeli będą się często zmieniały!", expanded=True):  # "expanded=False" sprawia, że sekcja jest zwinięta na początku
    st.dataframe(df.sample(x), use_container_width=True, hide_index=True)


# Wyświetlenie wyników w Streamlit PO FILTROWANIU
statystyki_plec_filtr = filtered_df["Płeć"].value_counts(dropna=False)
liczba_mezczyzn_filtr = statystyki_plec_filtr.get("Mężczyzna", 0)  # Wartość "Mężczyzna"
liczba_kobiet_filtr = statystyki_plec_filtr.get("Kobieta", 0)  # Wartość "Kobieta"
liczba_brak_podanej_pleci_filtr = statystyki_plec_filtr.get(np.nan, 0) + statystyki_plec_filtr.get("Inne", 0)

# Suma osób
suma_osob_filtr = liczba_mezczyzn_filtr + liczba_kobiet_filtr + liczba_brak_podanej_pleci_filtr

# Tworzenie układu kolumn
c0, c1, c2, c3 = st.columns(4)
with c0:
    st.metric("Liczba (po filtrowaniu)", suma_osob_filtr)  # Liczba wszystkich uczestników
with c1:
    st.metric("Liczba mężczyzn", liczba_mezczyzn_filtr)
with c2:
    st.metric("Liczba kobiet", liczba_kobiet_filtr)
with c3:
    st.metric("Brak podanej płci / inna", liczba_brak_podanej_pleci_filtr)

#
# Wyświetlanie przefiltrowanych danych
#
with st.expander("Zobacz przefiltrowane dane", expanded=False):  # "expanded=False" sprawia, że sekcja jest zwinięta na początku
    st.write("### Przefiltrowane dane")
    st.dataframe(filtered_df)

# Stworzenie rozwijanego paska, który zawiera kod rysowania wykresu
with st.expander("Zobacz wykres wybrany w pasku bocznym", expanded=True):  # "expanded=False" sprawia, że sekcja jest zwinięta na początku
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

# Stworzenie rozwijanego paska, który zawiera kod rysowania wykresu
with st.expander("Zobacz histogram z linią trendu - Przedział wiekowy / ilość", expanded=False):  # "expanded=False" sprawia, że sekcja jest zwinięta na początku
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

# Stworzenie rozwijanego paska dla wykresu
with st.expander("Zobacz wykres kołowy dla hobby", expanded=False):  # "expanded=False" sprawia, że sekcja jest zwinięta na początku
    rysuj_wykres_kolowy_hobby()

#
# SKRZYPCE
#

# Kolejność dla "lata doświadczenia"
experience_order = ['0-2', '3-5', '6-10', '11-15', '>=16']

# Stworzenie rozwijanego paska dla wykresu skrzypcowego
with st.expander("Zobacz wykres skrzypcowy: Lata doświadczenia vs. Ulubione miejsce", expanded=False):
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
# Stworzenie rozwijanego paska dla macierzy korelacji
with st.expander("Zobacz macierz korelacji", expanded=False):
    # Macierz korelacji
    st.write("## Macierz korelacji")
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Greens', cbar=True)
    st.pyplot(plt.gcf())
