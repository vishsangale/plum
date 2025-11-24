from pypdf import PdfReader

reader = PdfReader("2510.07784v1.pdf")
text = ""
print(f"Total pages: {len(reader.pages)}")
for i in range(3, 4):
    if i < len(reader.pages):
        print(f"--- Page {i} ---")
        print(reader.pages[i].extract_text())
        print("----------------")

print(text)
