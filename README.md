# NVT Projekat

**Student:** Sava Janjić SV51/2021

**Ciljna ocena:** 10

## K-Means

### Opis problema

K-Means je algoritam koji se koristi u mašinskom učenju za klasterovanje podataka. Zadatak ovog algoritma nenadgledanog učenja je da podatke razvrsta u klastere na osnovu njihovih osobina tako što inicijalno postavi centre klastera, a zatim za svaki podatak računa njegovu distancu do centra svakog klastera. Za računanje distance se koristi neka metrika udaljenosti kao što je Menhetn ili Euklidska udaljenost. Nakon dodele podataka klasterima, centri klastera se menjaju i algoritam se nastavlja dok ne konvergira.

Velika mana sekvencijalnog K-Means algoritma je brzina izvršavanja i vreme konvergencije kod velikih skupova podataka. Usko grlo je dodeljivanje centra klastera svakom podatku u skupu.

Vremenska složenost klasičnog sekvencijalnog pristupa: $O(I \times N \times K \times D)$, gde je $I$ maksimalni broj iteracija, $N$ broj podataka, $K$ broj klastera, a $D$ broj dimenzija. Ova složenost je neprihvatljiva za velike skupove podataka.

### Paralelizacija K-Means algoritma

Algoritam je pogodan za paralelizaciju jer za svaku tačku (podatak) u skupu možemo pronaći udaljenost od centara klastera nezavisno. Svakoj niti se može dodeliti tačka ili mali skup tačaka koji će obrađivati. Kada se pronađe kom klasteru pripada tačka, algoritam treba da doda koordinate te tačke i poveća brojač tog klastera. Atomskim operacijama se garantuje da će se sabiranje i brojanje izvesti ispravno, čak i kada više niti pokušava da istovremeno ažurira isti centroid klastera. 