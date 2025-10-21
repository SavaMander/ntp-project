import pandas as pd
import os
import glob

# Definišite putanju do direktorijuma sa fajlovima (na osnovu priložene slike)
data_dir = os.path.join('weak_scaling')

# Lista za čuvanje rezultata
results = []

# Koristite glob za pronalaženje svih CSV fajlova koji počinju sa 'kmeans_n'
# i završavaju sa '_results.csv' unutar definisanog direktorijuma
file_pattern = os.path.join(data_dir, 'kmeans_n*_results.csv')
print(file_pattern)
file_list = glob.glob(file_pattern)

print(f"Pronađeno je {len(file_list)} fajlova za obradu u: {data_dir}\n")

if not file_list:
    print("Greška: Nije pronađen nijedan fajl. Proverite putanju i format imena fajlova.")
else:
    # Prolazak kroz svaki pronađeni fajl
    for file_path in file_list:
        try:
            # Učitavanje CSV fajla u DataFrame
            # U CSV fajlu se koristi zarez kao separator (obično je to podrazumevano)
            df = pd.read_csv(file_path)

            # Provera da li kolona 'TotalTime_s' postoji
            if 'TotalTime_s' in df.columns:
                # Računanje standardne devijacije
                std_dev = df['TotalTime_s'].std()

                # Izdvajanje broja 'n' iz imena fajla (npr. iz 'kmeans_n10_results.csv' dobijamo 10)
                file_name = os.path.basename(file_path)
                # Pretpostavljamo da je broj n nakon 'kmeans_n' i pre '_results'
                try:
                    n_value = file_name.split('_n')[1].split('_results')[0]
                except IndexError:
                    n_value = file_name # Ako je format imena neispravan

                # Čuvanje rezultata
                results.append({
                    'File': file_name,
                    'N_Value': n_value,
                    'TotalTime_s_StdDev': std_dev
                })

                print(f"Obrada: {file_name} -> Standardna devijacija TotalTime_s: {std_dev:.6f}")
            else:
                print(f"Upozorenje: Kolona 'TotalTime_s' nije pronađena u fajlu: {file_path}")

        except Exception as e:
            print(f"Greška prilikom obrade fajla {file_path}: {e}")

# Prikaz svih rezultata u tabelarnom formatu
if results:
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("Konačni Rezultati Standardne Devijacije:")
    print("="*50)
    # Sortiranje radi bolje preglednosti
    results_df = results_df.sort_values(by='N_Value', key=lambda x: pd.to_numeric(x, errors='coerce'))
    print(results_df.to_markdown(index=False, floatfmt=".6f"))
    print("="*50)