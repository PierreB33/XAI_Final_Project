import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from PIL import Image


def create_spectrogram(audio_path, output_image_path, mel=True, n_mels=128, fmax=8000):
    """
    Convertit un fichier audio .wav en une image spectrogramme .png
    
    Args:
        audio_path (str): Chemin vers le fichier audio .wav
        output_image_path (str): Chemin pour sauvegarder le spectrogramme
        mel (bool): Utiliser Mel-spectrogram si True, sinon STFT
        n_mels (int): Nombre de bandes Mel
        fmax (int): Fréquence maximale
    """
    try:
        plt.figure(figsize=(10, 4))
        
        # Charger l'audio
        y, sr = librosa.load(audio_path, sr=None)
        
        if mel:
            # Générer le spectrogramme de Mel
            ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
        else:
            # Générer le STFT
            ms = np.abs(librosa.stft(y))
        
        # Convertir en dB
        log_ms = librosa.power_to_db(ms, ref=np.max)
        
        # Affichage sans bordures pour l'IA
        librosa.display.specshow(log_ms, sr=sr, fmax=fmax)
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
    except Exception as e:
        raise Exception(f"Erreur lors de la conversion audio: {str(e)}")


def create_multiple_spectrograms(audio_path, output_dir):
    """
    Créer plusieurs représentations spectrogrammes du même fichier audio
    
    Args:
        audio_path (str): Chemin vers le fichier audio
        output_dir (str): Répertoire de sortie
        
    Returns:
        dict: Chemins vers les spectrogrammes créés
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    specs = {}
    
    # Mel-spectrogram standard
    mel_path = os.path.join(output_dir, f"{base_name}_mel.png")
    create_spectrogram(audio_path, mel_path, mel=True)
    specs['mel'] = mel_path
    
    # STFT
    stft_path = os.path.join(output_dir, f"{base_name}_stft.png")
    create_spectrogram(audio_path, stft_path, mel=False)
    specs['stft'] = stft_path
    
    return specs


def batch_convert_wavs(input_folder, output_folder, recursive=True):
    """
    Convertir tous les fichiers .wav d'un dossier en spectrogrammes
    
    Args:
        input_folder (str): Dossier contenant les fichiers .wav
        output_folder (str): Dossier de sortie pour les images
        recursive (bool): Chercher récursivement dans les sous-dossiers
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    converted_count = 0
    
    if recursive:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.wav'):
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_folder, file.replace('.wav', '.png'))
                    try:
                        create_spectrogram(input_file, output_file)
                        converted_count += 1
                    except Exception as e:
                        print(f"Erreur pour {file}: {str(e)}")
    else:
        for file in os.listdir(input_folder):
            if file.endswith('.wav'):
                input_file = os.path.join(input_folder, file)
                output_file = os.path.join(output_folder, file.replace('.wav', '.png'))
                try:
                    create_spectrogram(input_file, output_file)
                    converted_count += 1
                except Exception as e:
                    print(f"Erreur pour {file}: {str(e)}")
    
    return converted_count


def get_audio_info(audio_path):
    """
    Obtenir des informations sur un fichier audio
    
    Args:
        audio_path (str): Chemin vers le fichier audio
        
    Returns:
        dict: Informations (durée, taux d'échantillonnage, etc.)
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'num_samples': len(y),
            'file_size': os.path.getsize(audio_path),
            'path': audio_path
        }
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture des informations audio: {str(e)}")
