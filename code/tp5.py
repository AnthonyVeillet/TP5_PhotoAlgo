# tp5.py
# Librairies nécessaires : torch, diffusers, transformers, accelerate, matplotlib, Pillow, time, os, platform
# pip install -r requirements.txt

import torch
import os
import time
import platform
import matplotlib.pyplot as plt


from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusion import generate_image


# Chemins vers les données
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMGS_DIR = os.path.join(BASE_DIR, "data", "dataInput", "imgs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "dataOutput")


def main() -> None:

    #torch.set_num_threads(8)

    # Charger un modèle de type Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-small-2m")

    # Déplacer sur GPU si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Texte pour la génération
    #prompt = "a tropical bird sitting on a branch of a tree"
    #prompt = "Un astronaute chevauchant un cheval sur Mars"
    prompt = "a horse on Mars"

    # Générateur aligné sur le device (CPU/CUDA) pour éviter les erreurs de device mismatch
    generator = torch.Generator(device=device).manual_seed(42)

    # Appeler notre boucle de diffusion minimale
    image = generate_image(
        pipe=pipe,
        prompt=prompt,
        num_inference_steps=10,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=generator,
    )

    image.save("example.png")

    # ============================================================
    # SECTION 1 — Réchauffement (15%)
    # Générer au moins 3 images avec des prompts différents.
    # Noter les spécifications système et le temps de génération.
    # ============================================================
    section1(pipe, device)

    # ============================================================
    # SECTION 2 — Guidage sans classifieur / CFG (15%)
    # Générer des images avec 12 valeurs de guidance_scale différentes.
    # ============================================================
    section2(pipe, device)

    # ============================================================
    # SECTION 3 — Validation de l'inpainting (30%)
    # Tester l'inpainting implémenté dans diffusion.py.
    # ============================================================
    section3(pipe, device)

    # ============================================================
    # SECTION 4 — Suppression d'objets (15%)
    # Utiliser l'inpainting pour supprimer des objets.
    # ============================================================
    section4(pipe, device)

    # ============================================================
    # SECTION 5 — Remplacement d'objets (15%)
    # Utiliser l'inpainting pour remplacer des objets.
    # ============================================================
    section5(pipe, device)


# ============================================================
# Fonctions utilitaires
# ============================================================

def make_generator(device: str, seed: int = 42) -> torch.Generator:
    # Crée un générateur avec une graine fixe pour la reproductibilité
    return torch.Generator(device=device).manual_seed(seed)


# ============================================================
# SECTION 1
# ============================================================
def section1(pipe, device: str):
    out_dir = os.path.join(OUTPUT_DIR, "Section 1")
    os.makedirs(out_dir, exist_ok=True)

    # Afficher les specs système
    print("=== Spécifications système ===")
    print(f"Processeur : {platform.processor() or platform.machine()}")
    print(f"Système    : {platform.system()} {platform.release()}")
    print(f"Device     : {device}")
    if device == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print()

    # Prompts utilisés pour la génération
    prompts = [
        "a beautiful sunset over the ocean with pink clouds",
        "a futuristic city under water with flying cars at night",
        "a cozy cabin in a snowy mountain forest",
    ]

    for i, prompt in enumerate(prompts):
        print(f"Génération image {i+1} : '{prompt}'")
        generator = make_generator(device)
        start = time.time()
        img = generate_image(
            pipe=pipe,
            prompt=prompt,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=generator,
        )
        elapsed = time.time() - start
        print(f"  Temps : {elapsed:.1f}s")
        img.save(os.path.join(out_dir, f"gen_{i+1}.png"))


# ============================================================
# SECTION 2
# ============================================================
def section2(pipe, device: str):
    out_dir = os.path.join(OUTPUT_DIR, "Section 2")
    os.makedirs(out_dir, exist_ok=True)

    prompt = "a unicorn on the Moon, photorealistic"
    scales = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 40.0, 60.0]

    images = []
    for s in scales:
        print(f"Guidance scale = {s}")
        generator = make_generator(device)
        img = generate_image(
            pipe=pipe,
            prompt=prompt,
            num_inference_steps=10,
            guidance_scale=s,
            height=512,
            width=512,
            generator=generator,
        )
        img.save(os.path.join(out_dir, f"cfg_{s}.png"))
        images.append((s, img))

    # Créer la grille comparative
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Classifier-Free Guidance — prompt: "{prompt}"', fontsize=14)
    for ax, (s, img) in zip(axes.flat, images):
        ax.imshow(img)
        ax.set_title(f"scale = {s}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cfg_grid.png"), dpi=150)
    plt.close()
    print("Grille CFG sauvegardée.")


# ============================================================
# SECTION 3
# ============================================================
def section3(pipe, device: str):
    out_dir = os.path.join(OUTPUT_DIR, "Section 3")
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(IMGS_DIR, "chateau_frontenac.png")
    mask_path = os.path.join(IMGS_DIR, "chateau_frontenac_mask.png")

    if not os.path.exists(img_path):
        print(f"Image introuvable : {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    prompt = "a beautiful castle courtyard, historic architecture"
    generator = make_generator(device)

    print("Test inpainting sur chateau_frontenac...")
    result = generate_image(
        pipe=pipe,
        prompt=prompt,
        num_inference_steps=10,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=generator,
        image=image,
        mask=mask,
    )
    result.save(os.path.join(out_dir, "inpainting_test.png"))
    print("Validation inpainting terminée.")


# ============================================================
# SECTION 4
# ============================================================
def section4(pipe, device: str):
    out_dir = os.path.join(OUTPUT_DIR, "Section 4")
    os.makedirs(out_dir, exist_ok=True)

    cases = [
        ("graffiti.png", "graffiti_mask.png", "a clean brick wall"),
        ("pisa.png", "pisa_mask.png", "a green field with blue sky and buildings"),
        ("chateau_frontenac.png", "chateau_frontenac_mask.png",
        "a cobblestone street in front of a historic castle"),
        # Cas personnel ici (image perso + masque perso)
        ("mon_image1.png", "mon_image1_mask.png", "a bar table with society games on a shelf in the background"),
    ]

    for img_name, mask_name, prompt in cases:
        img_path = os.path.join(IMGS_DIR, img_name)
        mask_path = os.path.join(IMGS_DIR, mask_name)

        if not os.path.exists(img_path):
            print(f"  Image introuvable : {img_path}, on saute.")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        generator = make_generator(device)

        print(f"Suppression sur {img_name} — prompt: '{prompt}'")
        result = generate_image(
            pipe=pipe,
            prompt=prompt,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=generator,
            image=image,
            mask=mask,
        )
        base = os.path.splitext(img_name)[0]
        result.save(os.path.join(out_dir, f"{base}_result.png"))

    print("Section 4 terminée.")


# ============================================================
# SECTION 5
# ============================================================
def section5(pipe, device: str):
    out_dir = os.path.join(OUTPUT_DIR, "Section 5")
    os.makedirs(out_dir, exist_ok=True)

    cases = [
        ("pyramides.webp", "pyramides_mask.png", "snow covered pyramid in a winter desert"),
        ("baie_beauport.avif", "baie_beauport_mask.png", "a penguin standing on a sandy beach"),
        ("moulin.png", "moulin_mask.png", "a giant standing in a field"),
        # Cas personnel ici (image perso + masque perso)
        ("mon_image2.png", "mon_image2_mask.png", "a road going to an ice castle in a snowy landscape with mountains in the background"),
    ]

    for img_name, mask_name, prompt in cases:
        img_path = os.path.join(IMGS_DIR, img_name)
        mask_path = os.path.join(IMGS_DIR, mask_name)

        if not os.path.exists(img_path):
            print(f"  Image introuvable : {img_path}, on saute.")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        generator = make_generator(device)

        print(f"Remplacement sur {img_name} — prompt: '{prompt}'")
        result = generate_image(
            pipe=pipe,
            prompt=prompt,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=512,
            width=512,
            generator=generator,
            image=image,
            mask=mask,
        )
        base = os.path.splitext(img_name)[0]
        result.save(os.path.join(out_dir, f"{base}_result.png"))

    print("Section 5 terminée.")


if __name__ == "__main__":
    main()