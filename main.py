import tkinter as tk
from tkinter import messagebox

from ui.app import QiFlowApp
from utils.deps import check_dependencies


def main() -> None:
    missing_core, missing_optional = check_dependencies()
    if missing_core:
        root = tk.Tk()
        root.withdraw()
        message = "Missing core dependencies: " + ", ".join(missing_core) + "\nRun: pip install -r requirements.txt"
        messagebox.showerror("QiFlow Dependencies", message)
        root.destroy()
        return
    if missing_optional:
        root = tk.Tk()
        root.withdraw()
        message = "Optional dependencies missing: " + ", ".join(missing_optional) + "\nSome features will be disabled."
        messagebox.showwarning("QiFlow Dependencies", message)
        root.destroy()
    app = QiFlowApp()
    app.run()


if __name__ == "__main__":
    main()
