# NTP Projekat

**Student:** Sava Janjić SV51/2021

**Ciljna ocena:** 10

## K-Means

### Opis problema

K-Means je algoritam koji se koristi u mašinskom učenju za klasterovanje podataka. Zadatak ovog algoritma nenadgledanog učenja je da podatke razvrsta u klastere na osnovu njihovih osobina tako što inicijalno postavi centre klastera, a zatim za svaki podatak računa njegovu distancu do centra svakog klastera. Za računanje distance se koristi neka metrika udaljenosti kao što je Menhetn ili Euklidska udaljenost. Nakon dodele podataka klasterima, centri klastera se menjaju i algoritam se nastavlja dok ne konvergira.

Velika mana sekvencijalnog K-Means algoritma je brzina izvršavanja i vreme konvergencije kod velikih skupova podataka. Usko grlo je dodeljivanje centra klastera svakom podatku u skupu.

Vremenska složenost klasičnog sekvencijalnog pristupa: $O(I \times N \times K \times D)$, gde je $I$ maksimalni broj iteracija, $N$ broj podataka, $K$ broj klastera, a $D$ broj dimenzija. Ova složenost je neprihvatljiva za velike skupove podataka.

### Paralelizacija K-Means algoritma

Algoritam je pogodan za paralelizaciju jer za svaku tačku (podatak) u skupu možemo pronaći udaljenost od centara klastera nezavisno. Svakoj niti se može dodeliti skup tačaka koji će obrađivati. Kada se za svaki klaster pronadju tačke koje mu pripadaju, potrebno je ažurirati centre tog klastera. To se može uraditi tako što će se za svaki klaster kreirati nit koja će izračunati novu poziciju centra na osnovu podataka koji pripadaju tom klasteru. 

### Ocena 6

**Sekvencijalna verzija**

Implementacija osnovnog K-Means algoritma korišćenjem numpy za proračun, pandas za učitavanje fajlova i sklearn za skaliranje podataka. U svakoj iteraciji postoje dve faze. E-korak za svaku tačku pronalazi najbliži centar, dok u sledećem M-koraku se za svaki klaster pronalazi novi centar.

**Paralelizovana verzija**

Implementacija K-Means algoritma uz pomoć `multiprocessing` biblioteke za rad sa više niti. U E-koraku svaka nit obrađuje deo podataka (chunk) i za svaki podatak određuje njemu najbliži centar klastera Euklidskom distancom. Nakon toga, u M-koraku svakoj niti se dodeljuje po jedan klaster (ili više ako je broj klastera veći od broja niti) i za njega se pronalazi novi centar. Algoritam se nastavlja dok ne konvergira ili dok se ne izvrši maksimalan broj iteracija. Za paralelizovanu verziju su takođe korišćene biblioteke iz sekvencijalne verzije.

### Ocena 7 

**Sekvencijalna verzija**

Implementacija osnovnog K-Means algoritma u Rust-u korišćenjem biblioteka ndarray, rand (seed je uvek isti) i csv.

**Paralelizovana verzija**

Implementacija paralelizovanog K-Means algoritma u Rust-u je slična kao i u Python-u uz oslonac na `thread` biblioteku.

### Ocene 8 i 9
Odrediće se procenat sekvencijalnog dela i dela koda koji je moguće paralelizovati. Na osnovu tih parametara možemo odrediti teorijske maksimume u skladu sa Amdalovim, odnosno Gustafsonovim zakonom. Izvršiće se eksperimenti jakog i slabog skaliranja i rezultati će se prikazati na graficima (takođe će biti upisani u fajlove), a opažanja će biti zapisana u izveštaju. Za svaki broj niti algoritam će biti pokrenut 30 puta kako bi dobili relevantno vreme izvršenja tj. prosek svih vremena izvršavanja.

### Ocena 10
Uz pomoć biblioteke Plotters biće prikazana promena pozicija centara klastera po svakoj iteraciji na osnovu učitanog skupa podataka.