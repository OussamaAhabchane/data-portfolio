#!/usr/bin/env python3
"""
Script d'installation et setup du portfolio Data Science
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Affiche un header stylisé"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_command(command, description):
    """Exécute une commande avec gestion d'erreur"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"  Output: {e.output}")
        return False


def check_python_version():
    """Vérifie la version de Python"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    else:
        print("✓ Python version is compatible")


def create_virtual_environment():
    """Crée un environnement virtuel"""
    print_header("Creating Virtual Environment")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("ℹ️  Virtual environment already exists")
        response = input("Do you want to recreate it? (y/n): ")
        if response.lower() != 'y':
            return
    
    return run_command(
        f"{sys.executable} -m venv venv",
        "Creating virtual environment"
    )


def install_requirements():
    """Installe les requirements"""
    print_header("Installing Requirements")
    
    # Détecter le chemin de pip dans le venv
    if sys.platform == "win32":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    # Upgrade pip
    run_command(
        f"{pip_path} install --upgrade pip",
        "Upgrading pip"
    )
    
    # Installer les requirements globaux
    success = run_command(
        f"{pip_path} install -r requirements.txt",
        "Installing global requirements"
    )
    
    return success


def download_nltk_data():
    """Télécharge les ressources NLTK"""
    print_header("Downloading NLTK Data")
    
    try:
        import nltk
        print("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False


def download_spacy_model():
    """Télécharge le modèle spaCy"""
    print_header("Downloading spaCy Model")
    
    if sys.platform == "win32":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    return run_command(
        f"{python_path} -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )


def create_directory_structure():
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    print_header("Creating Directory Structure")
    
    directories = [
        "01-stock-sentiment-prediction/data/raw",
        "01-stock-sentiment-prediction/data/processed",
        "01-stock-sentiment-prediction/visualizations",
        "02-fraud-detection/data/raw",
        "02-fraud-detection/data/processed",
        "02-fraud-detection/models",
        "03-ecommerce-review-analysis/data/raw",
        "03-ecommerce-review-analysis/data/processed",
        "03-ecommerce-review-analysis/models",
        "04-customer-churn-prediction/data/raw",
        "04-customer-churn-prediction/data/processed",
        "04-customer-churn-prediction/models",
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            # Créer un .gitkeep pour garder le dossier dans git
            (path / ".gitkeep").touch()
    
    print("✓ Directory structure created")


def print_next_steps():
    """Affiche les prochaines étapes"""
    print_header("Setup Complete! 🎉")
    
    print("Next steps:")
    print("\n1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Choose a project and download data:")
    print("   cd 01-stock-sentiment-prediction")
    print("   python data/download_data.py")
    
    print("\n3. Open Jupyter notebooks:")
    print("   jupyter notebook notebooks/")
    
    print("\n4. Or run the main script:")
    print("   python src/main.py")
    
    print("\n📚 Documentation:")
    print("   - Main README: README.md")
    print("   - Project READMEs: <project>/README.md")
    
    print("\n🔗 Useful links:")
    print("   - Kaggle datasets: https://www.kaggle.com/datasets")
    print("   - Yahoo Finance: https://finance.yahoo.com")
    
    print("\n💡 Tips:")
    print("   - Each project has its own requirements.txt")
    print("   - Check the project README for dataset links")
    print("   - Notebooks are in notebooks/ directories")


def main():
    """Fonction principale"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║        📊 Data Science Portfolio - Setup Script 📊       ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Vérifier Python
    check_python_version()
    
    # Créer l'environnement virtuel
    create_virtual_environment()
    
    # Installer les requirements
    if not install_requirements():
        print("\n⚠️  Some packages failed to install. Continue anyway? (y/n)")
        if input().lower() != 'y':
            sys.exit(1)
    
    # Télécharger NLTK data
    download_nltk_data()
    
    # Télécharger spaCy model
    response = input("\nDownload spaCy model? (required for NLP project) (y/n): ")
    if response.lower() == 'y':
        download_spacy_model()
    
    # Créer la structure
    create_directory_structure()
    
    # Afficher les prochaines étapes
    print_next_steps()


if __name__ == "__main__":
    main()
