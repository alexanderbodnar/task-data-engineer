Celé riešenie úlohy sa nachádza v súbore solution.py. 

Hlavné časti pre zbehnutie modelu a evaluáciu výsledkov som zobral z main.py

Funkcia na predspracovanie raw datasetu:
    1. Odstráni column AGENCY (nie je potrebný, keďže je unseen v modeli)
    2. Odstránenie všetkých NaN hodnot (dopočítanie pomocou medianu)
    3. V datasete sa nachádzali aj Null stringy. Zmenená na NaN, ktoré sa následne dajú tiež zmeniť na medianu
    4. Premenovanie stlpcov na názvy ktore boli použité pri trenovani modelu
    5. Encodovanie label stlpca 'ocean_proximity' na 0,1 (teda vytvorenie aj jednotlivych novych stlpcov)


Pre ukladanie df som si vybral PostgreSQL.

PRE FUNGOVANIE KÓDU JE NUTNÉ MAŤ POSTGRESQL a zmeniť DATABASE_URL