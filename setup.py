#!/usr/bin/env python
"""
Setup script for Gender Prediction App
"""
import os
import shutil
import argparse

def create_directory_structure():
    """Create the necessary directory structure for the project"""
    print("Creating directory structure...")
    
    # Create main directories
    directories = [
        "data",
        "models/lr_model",
        "models/nb_model",
        "utils",
        "static"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True

def create_empty_init_files():
    """Create empty __init__.py files in the utils directory"""
    print("Creating __init__.py files...")
    
    init_file_path = os.path.join("utils", "__init__.py")
    with open(init_file_path, "w") as f:
        f.write('"""Utility modules for Gender Prediction App"""\n')
    
    print(f"Created file: {init_file_path}")
    
    return True

def copy_demo_data():
    """Create a demo data file for testing"""
    print("Creating demo data file...")
    
    demo_data = """NOMPL;SEXE
Fatimetou Ahmed Mbareck;F
Mariem Tah Mohamed Elmoktar Essaghir;F
Aicha Ahmed Beyhime;F
Nanna El Mounir Hame;F
Vatme Mohamed Magha;F
Fatimtou Zahra Ely Mohamed Lmin;F
El Hadj Samba Abou Diop;M
Zeinebou Ahmedou Akembi;F
Mama Mohamed Sidi M'Hamed;F
Bouye Ahmed Ahmed Djoume;M
Bouna Itawol Amrou Abdoullah;M
Lamine Amadou Ba;M
Halima Esghaier Mbarek;M
Vatma El Hasniya Saad Bouh Hamady;F
Aichete Mahfoudh Khlil;F
Mouadh Mahfoudh Ekhlil;M
Zeinabou Mohamed El Mostapha Abdallahi;F
Ayoub Mahfoudh Ekhlil;M
Mariem Mohamed Mahmoud Taleb Sid'Ahmed;F
Abderrahmane Mohamed Ahmed Sghair;M
Mohamed Vadel Mohamed Boide;M
Mariem Badi Bady;F"""
    
    demo_file_path = os.path.join("data", "demo_data.csv")
    with open(demo_file_path, "w") as f:
        f.write(demo_data)
    
    print(f"Created demo data file: {demo_file_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup Gender Prediction App project structure")
    parser.add_argument("--clean", action="store_true", help="Clean existing directories before setup")
    
    args = parser.parse_args()
    
    if args.clean:
        print("Cleaning existing directories...")
        directories_to_clean = ["data", "models", "utils", "static"]
        for directory in directories_to_clean:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Cleaned directory: {directory}")
    
    # Create directory structure
    create_directory_structure()
    
    # Create empty __init__.py files
    create_empty_init_files()
    
    # Create demo data
    copy_demo_data()
    
    print("\nSetup completed successfully!")
    print("Run the app with: streamlit run app.py")

if __name__ == "__main__":
    main()