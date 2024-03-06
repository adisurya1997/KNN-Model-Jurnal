# import pandas as pd

# # Path ke file Excel (.xlsx)
# excel_file_path = "/Users/adisuryabasri/Downloads/data prakiraan/Prakiraan-Angin-Maksimum-Harian.xlsx"

# # Baca file Excel menjadi DataFrame
# df = pd.read_excel(excel_file_path)

# # Path untuk menyimpan file CSV
# csv_file_path = "/Users/adisuryabasri/Downloads/data prakiraan/Prakiraan-Angin-Maksimum-Harian.csv"

# # Tulis DataFrame ke file CSV
# df.to_csv(csv_file_path, index=False)

# print("File CSV telah berhasil dibuat:", csv_file_path)

import csv

# Path ke file CSV
csv_file_path = "/Users/adisuryabasri/Downloads/data prakiraan/Prakiraan-Angin-Maksimum-Harian.csv"

# Baris header baru
new_header = ["Lon","Lat","000","001","002","003","004","005","006","007","008","009","010","011","012","013","014","015","016","017","018","019","020","021","022","023","024","025","026","027","028","029","030","031","032","033","034","035","036","037","038","039","040","041","042","043","044","045","046","047","048","049","050","051","052","053","054","055","056","057","058","059","060","061","062","063","064","065","066","067","068","069","070","071","072","073","074","075","076","077","078","079","080","081","082","083","084","085","086","087","088","089","090","091","092","093","094","095","096","097","098","099",100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,234,237,240,243,246,249,252,255,258,261,264,267,270,273,276,279,282,285,288,291,294,297,300,303,306,309,312,315,318,321,324,327,330,333,336,339,342,345,348,351,354,357,360,363,366,369,372,375,378,381,384]

# Membaca file CSV
with open(csv_file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = [row for row in reader]

# Mengganti header
data[0] = new_header

# Menyimpan data kembali ke file CSV
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

print("Header file CSV berhasil diganti.")