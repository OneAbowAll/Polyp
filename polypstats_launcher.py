import os
import subprocess
import tkinter as tk
from tkinter import filedialog
#main_path 
#imgs_path 
#masks_path 
#mesh_name

def open_dir_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter main window
    folder_path = filedialog.askdirectory(title="Select a folder")
    root.destroy()
    return folder_path

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter main window
    file_path = filedialog.askopenfilename(title="Select a ply mesh")
    root.destroy()
    return file_path

def main():
    print("Current directory:", os.getcwd())
    main_path = ''
    imgs_path = ''
    masks_path = ''
    mesh_name =  ''
    metashape_name = ''
    imgs_path_FLUO = ''
    metashape_name_FLUO = ''
    transf_FLUO_RGB = ''

    try:
        with open("last.txt", "r") as f:
            content = f.read()
            print("Raw content:", repr(content))

            lines = content.splitlines()
            print("lines:", lines)
            if len(lines) >= 5:
                main_path = lines[0]
                imgs_path = lines[1]
                masks_path = lines[2]
                mesh_name = lines[3]
                metashape_name = lines[4]
                if len(lines) == 8:
                    imgs_path_FLUO = lines[5]
                    metashape_name_FLUO = lines[6]
                    transf_FLUO_RGB = lines[7]
            else:
                print("last.txt does not contain enough lines.")
    except FileNotFoundError:
        print("last.txt not found. Using default empty paths.")
        main_path = ""
        imgs_path = ""
        masks_path = ""
        mesh_name = ""
        metashape_name = ""
        imgs_path_FLUO = ""
        metashape_name_FLUO = ""
        transf_FLUO_RGB = ""

    
    #Ask for file paths----------------------------------------------------------------
    # Show a widget with the selected paths
    root = tk.Tk()
    root.title("Paths for the dataset")

    # Use StringVars to allow dynamic updates
    main_path_var = tk.StringVar(value=main_path)
    imgs_path_var = tk.StringVar(value=imgs_path)
    masks_path_var = tk.StringVar(value=masks_path)
    mesh_name_var = tk.StringVar(value=mesh_name)
    metashape_name_var = tk.StringVar(value=metashape_name)
    imgs_path_FLUO_var = tk.StringVar(value=imgs_path_FLUO)
    metashape_name_FLUO_var = tk.StringVar(value=metashape_name_FLUO)
    transf_FLUO_RGB_var = tk.StringVar(value=transf_FLUO_RGB)


    def set_main_path():
        path = open_dir_dialog()
        if path:
            main_path_var.set(path)

    def set_imgs_path():
        path = open_dir_dialog()
        if path:
            imgs_path_var.set(path)

    def set_masks_path():
        path = open_dir_dialog()
        if path:
            masks_path_var.set(path)

    def set_mesh_name():
        path = open_file_dialog()
        if path:
            mesh_name_var.set(path)

    def set_metashape_name():
        path = open_file_dialog()
        if path:
            metashape_name_var.set(path)

    def set_imgs_path_FLUO():
        path = open_dir_dialog()
        if path:
            imgs_path_FLUO_var.set(path)

    def set_metashape_name_FLUO():
        path = open_file_dialog()
        if path:
            metashape_name_FLUO_var.set(path)
    
    def set_transf_FLUO_RGB():
        path = open_file_dialog()
        if path:
            transf_FLUO_RGB_var.set(path)


    labels = [
        ("Images Path:", imgs_path_var, set_imgs_path),
        ("Mesh Name:", mesh_name_var, set_mesh_name),
        ("Metashape Name:", metashape_name_var, set_metashape_name),
        ("Ouput Path:", main_path_var, set_main_path),
        ("Cache Path:", masks_path_var, set_masks_path),
        ("Images Path FLUO:", imgs_path_FLUO_var, set_imgs_path_FLUO),
        ("Metashape Name FLUO:", metashape_name_FLUO_var, set_metashape_name_FLUO),
        ("Transformation FLUO to RGB:", transf_FLUO_RGB_var, set_transf_FLUO_RGB)       
    ]

    for i, (label_text, var, btn_cmd) in enumerate(labels):
        tk.Label(root, text=label_text).grid(row=i, column=0, sticky="e")
        tk.Entry(root, textvariable=var, width=50).grid(row=i, column=1, sticky="w")
        #tk.Label(root, textvariable=var).grid(row=i, column=1, sticky="w")
        tk.Button(root, text="Select", command=btn_cmd).grid(row=i, column=2, padx=5)

    def on_ok():
        with open("last.txt", "w") as f:
            f.write(main_path_var.get() + "\n")
            f.write(imgs_path_var.get() + "\n")
            f.write(masks_path_var.get() + "\n")
            f.write(mesh_name_var.get() + "\n")
            f.write(metashape_name_var.get() + "\n")
            f.write(imgs_path_FLUO_var.get() + "\n")
            f.write(metashape_name_FLUO_var.get() + "\n")
            f.write(transf_FLUO_RGB_var.get() + "\n")
            subprocess.Popen([
                "python", "polypstats.py",
                main_path_var.get(),
                imgs_path_var.get(),
                masks_path_var.get(),
                mesh_name_var.get(),
                metashape_name_var.get(),
                imgs_path_FLUO_var.get(),
                metashape_name_FLUO_var.get(),
                transf_FLUO_RGB_var.get()
            ])

    tk.Button(root, text="Launch !", command=on_ok).grid(row=len(labels), columnspan=3)
    root.mainloop()

    # Assign the selected values back to the variables
    main_path = main_path_var.get()
    imgs_path = imgs_path_var.get()
    masks_path = masks_path_var.get()
    mesh_name = mesh_name_var.get()
    metashape_name = metashape_name_var.get()
    imgs_path_FLUO = imgs_path_FLUO_var.get()
    metashape_name_FLUO = metashape_name_FLUO_var.get()
    transf_FLUO_RGB = transf_FLUO_RGB_var.get()


if __name__ == "__main__":
    main()