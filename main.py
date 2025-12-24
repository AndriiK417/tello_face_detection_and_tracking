import tkinter as tk
from gui import MainWindow

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    
    # Коректне закриття програми при натисканні на хрестик
    root.protocol("WM_DELETE_WINDOW", app.close)
    
    root.mainloop()
