import sys
import threading
import torch
import tkinter as tk

from Class.RAGPipeLine import RAGPipeLine


def select_device():
    device = torch.device("cpu")
    if torch.mps.is_available():
        device = torch.device("mps")
        print("MPS is available! Using MPS.")
    elif torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9

        print(f"Device ID: {device_id}")
        print(f"Device Name: {device_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print("GPU is available! Using GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")

    #device = torch.device("cpu")
    return device

def process_input(terminal, entry, rag, event=None):
    line = entry.get().strip()
    if not line:
        return
    terminal.configure(state=tk.NORMAL)
    terminal.insert(tk.END, f"You>> {line}\n")
    terminal.insert(tk.END, "Bot>> Thinking...\n")
    terminal.config(state=tk.DISABLED)
    terminal.see("end")
    entry.delete(0, tk.END)

    def run_query():
        try:
            response = rag.query(line)
        except Exception as ex:
            response = f"Error: {ex}"
        terminal.after(0, update_terminal, response)

    def update_terminal(response):
        terminal.configure(state=tk.NORMAL)
        terminal.delete("end-2l", "end-1l")
        terminal.insert(tk.END, f"Bot>> {response}\n\n")
        terminal.config(state=tk.DISABLED)
        terminal.see("end")
    threading.Thread(target=run_query, daemon=True).start()

def configure_input_window(rag):
    window = tk.Tk()
    window.title("ChatBot")
    window.geometry("1200x600")
    window.configure(background="black")

    terminal = tk.Text(window, font=("Arial", 12), state=tk.DISABLED, bg="black", fg="white", insertbackground="white")
    terminal.pack(fill="both", expand=True, padx=5, pady=5)
    terminal.configure(state=tk.NORMAL)
    terminal.insert(tk.END, "Bot: Ready. Ask me anything.\n\n")
    terminal.configure(state=tk.DISABLED)

    entry = tk.Entry(window, font=("Arial", 12), bg="#333", fg="white", insertbackground="white")
    entry.pack(fill=tk.X, padx=5, pady=5)
    entry.bind("<Return>", lambda event: process_input(terminal, entry, rag, event))
    entry.focus()

    window.mainloop()


def main():
    print(sys.version)
    print(sys.executable)
    document = "quran-english.pdf"
    rag = RAGPipeLine(select_device(), document.replace(".pdf", ""))
    rag.load_pdf(document)

    configure_input_window(rag)


if __name__ == '__main__':
    main()