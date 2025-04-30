# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import traceback

class PostureGUI:
    """Handles the real-time display GUI using Tkinter."""
    def __init__(self, root):
        self.root = root
        self.root.title("MSD Posture Analysis")
        # Handle window close button press
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.is_closing = False # Flag to signal closing process

        # --- Main Frame ---
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure resizing behavior
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1) # Allow content to expand horizontally

        # --- Widgets ---
        self.label = ttk.Label(self.frame, text="Real-Time Analysis", font=("Arial", 16))
        self.label.grid(row=0, column=0, columnspan=2, pady=5, sticky='ew')

        self.person_id_var = tk.StringVar(value="Person: N/A")
        self.person_id_label = ttk.Label(self.frame, textvariable=self.person_id_var, font=("Arial", 12))
        self.person_id_label.grid(row=1, column=0, columnspan=2, pady=2, sticky='w') # Align left

        self.status_var = tk.StringVar(value="Status: N/A")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var, font=("Arial", 12))
        self.status_label.grid(row=2, column=0, columnspan=2, pady=2, sticky='w') # Align left

        self.score_var = tk.StringVar(value="Score: N/A")
        self.score_label = ttk.Label(self.frame, textvariable=self.score_var, font=("Arial", 10))
        self.score_label.grid(row=3, column=0, columnspan=2, pady=2, sticky='w') # Align left

        # Suggestions Area
        self.suggestion_label_title = ttk.Label(self.frame, text="Suggestions:", font=("Arial", 10, "bold"))
        self.suggestion_label_title.grid(row=4, column=0, columnspan=2, pady=(5,0), sticky='w')

        # Use a Text widget for multi-line suggestions
        self.suggestion_text = tk.Text(self.frame, height=4, wrap=tk.WORD, font=("Arial", 9),
                                       relief="solid", borderwidth=1, state=tk.DISABLED) # Start disabled
        self.suggestion_text.grid(row=5, column=0, columnspan=2, pady=2, sticky='ew')
        # Add scrollbar if needed (optional)
        # scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.suggestion_text.yview)
        # scrollbar.grid(row=5, column=2, sticky='ns')
        # self.suggestion_text['yscrollcommand'] = scrollbar.set

        # --- Separator and Quit Button ---
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).grid(row=6, column=0, columnspan=2, sticky='ew', pady=10)

        self.quit_button = ttk.Button(self.frame, text="Stop Analysis & Quit", command=self.on_closing)
        self.quit_button.grid(row=7, column=0, columnspan=2, pady=5)

    def update(self, person_id, status, score, suggestions):
        """Updates the GUI labels with the latest analysis results."""
        if self.is_closing: return # Don't update if closing

        try:
            self.person_id_var.set(f"Person: {person_id}" if person_id is not None else "Person: N/A")
            self.status_var.set(f"Status: {status if status else 'N/A'}")
            self.score_var.set(f"Score: {score if score is not None and score != -1 else 'N/A'}")

            # Update suggestions text widget
            self.suggestion_text.config(state=tk.NORMAL) # Enable editing
            self.suggestion_text.delete("1.0", tk.END) # Clear previous text
            if suggestions:
                self.suggestion_text.insert(tk.END, "\n".join(suggestions))
            else:
                 self.suggestion_text.insert(tk.END, "N/A")
            self.suggestion_text.config(state=tk.DISABLED) # Disable editing

            # Use update_idletasks for potentially smoother updates within a loop
            self.root.update_idletasks()

        except tk.TclError as e:
            # Handle cases where the GUI might be destroyed during update
            if "application has been destroyed" not in str(e).lower():
                 print(f"Error updating GUI: {e}")
                 print(traceback.format_exc())
            # If destroyed, set flag to prevent further updates
            self.is_closing = True
        except Exception as e:
            print(f"Unexpected error updating GUI: {e}")
            print(traceback.format_exc())
            self.is_closing = True # Stop updates on unexpected error

    def on_closing(self):
        """Handles window close event (button press or 'X')."""
        if self.is_closing: return # Avoid multiple calls

        print("GUI closing requested.")
        # Optional: Confirmation dialog
        # if messagebox.askokcancel("Quit", "Stop analysis and close?"):
        self.is_closing = True # Signal that closing is in progress
        try:
            # Destroy the Tkinter window
            # Need to be careful here, destroying root might abruptly end things
            # It's better if the main loop checks self.is_closing and exits gracefully.
            # self.root.quit() # Stops the Tkinter mainloop if it's running independently
            self.root.destroy() # Destroys the window and widgets
            print("GUI window destroyed.")
        except tk.TclError:
            print("GUI already destroyed.") # Ignore if already destroyed
        except Exception as e:
            print(f"Error destroying GUI root: {e}")
        # else:
        #     print("Quit cancelled.") # If using confirmation dialog