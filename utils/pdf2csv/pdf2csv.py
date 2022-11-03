import tabula

input_file_path = "/home/mhs/Downloads/Documents/1712.03704.pdf"
page_number = [19, 20, 22, 23]

for i in page_number:
    df = tabula.read_pdf(input_file_path, pages=i)[0]
    df.to_csv(f"./output_{i}.csv")
