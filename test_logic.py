from logic import process_text

tests = [
    "AAPL 1000",
    "MSFT 2500",
    "AAPL abc",
    "AAPL -10",
    "AAAAAA 1000",   # вероятно несуществующий тикер
    "AAPL",          # неправильный формат
]

for t in tests:
    print("INPUT:", t)
    res = process_text(t)
    print("MESSAGE:\n", res["message"])
    print("-" * 40)
