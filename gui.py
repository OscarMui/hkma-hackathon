import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import re

# Function to run the chosen model script
def run_script():
    base_value = base_entry.get()
    if not re.match(r'^[0-9]+$', base_value):
        output_text.insert(tk.END, "Invalid base value. Please enter a valid number.\n")
        return

    if model_var.get() == "Linear Regression":
        script_path = '07310929LR.py'
    else:
        script_path = '07310929SVM.py'

    # Read the script with explicit encoding
    try:
        with open(script_path, 'r', encoding='utf-8') as file:
            script_content = file.read()
    except UnicodeDecodeError as e:
        output_text.insert(tk.END, ".")
        return

    # Replace "base" with the user-provided base value
    updated_script = re.sub(r'base\s*=\s*[0-9]+', f'base = {base_value}', script_content)

    # Write the updated script to a temporary file with explicit encoding
    temp_script_path = 'temp_script.py'
    try:
        with open(temp_script_path, 'w', encoding='utf-8') as file:
            file.write(updated_script)
    except Exception as e:
        output_text.insert(tk.END, ".")
        return

    # Run the script and capture the output
    try:
        result = subprocess.run(['python', temp_script_path], capture_output=True, text=True)
        output_text.insert(tk.END, result.stdout)
        if result.stderr:
            output_text.insert(tk.END, ".")
    except Exception as e:
        output_text.insert(tk.END, ".")

    # Display the plot if Linear Regression was chosen
    if model_var.get() == "Linear Regression":
        try:
            # The plot is assumed to be saved by the script as 'plot.png'
            plot_image = tk.PhotoImage(file='plot.png')
            plot_label.configure(image=plot_image)
            plot_label.image = plot_image
        except Exception as e:
            output_text.insert(tk.END, ".")

# Create the main window
root = tk.Tk()
root.title("Model Runner")

# Base input
ttk.Label(root, text="Enter base value:").grid(row=0, column=0, padx=10, pady=10)
base_entry = ttk.Entry(root)
base_entry.grid(row=0, column=1, padx=10, pady=10)

# Model selection
model_var = tk.StringVar(value="Linear Regression")
ttk.Radiobutton(root, text="Linear Regression", variable=model_var, value="Linear Regression").grid(row=1, column=0, padx=10, pady=10)
ttk.Radiobutton(root, text="Support Vector Machine", variable=model_var, value="Support Vector Machine").grid(row=1, column=1, padx=10, pady=10)

# Run button
run_button = ttk.Button(root, text="Run", command=run_script)
run_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Output text box
output_text = scrolledtext.ScrolledText(root, width=80, height=20)
output_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Plot label
plot_label = ttk.Label(root)
plot_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# Start the main event loop
root.mainloop()
