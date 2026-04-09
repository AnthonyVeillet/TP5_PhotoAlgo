# diffusion.py
# Librairies nécessaires : torch, tqdm, numpy, Pillow
# pip install torch tqdm numpy Pillow

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


@torch.no_grad()
def generate_image(
    pipe,
    prompt: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    generator: torch.Generator | None = None,
    image: Image.Image | None = None,
    mask: Image.Image | None = None,
):
    """
    Boucle de diffusion avec support optionnel d'inpainting.
    Si image et mask sont fournis, effectue l'inpainting.
    Le masque doit être blanc (255) là où on veut régénérer, noir (0) là où on conserve.
    """
    device = pipe.device

    # 1. Encoder la description textuelle (prompt) en embeddings
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Embeddings inconditionnels pour le guidage sans classifieur (classifier-free guidance)
    uncond_input = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    # Concaténer les embeddings conditionnels et inconditionnels
    prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])

    # 2. Préparer les pas de temps (timesteps)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 3. Échantillonner le bruit gaussien initial (latents)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=text_embeddings.dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    # --- Préparation inpainting (si image et mask fournis) --- DEBUT AJOUT V1
    inpainting_mode = image is not None and mask is not None
    if inpainting_mode:
        # Redimensionner image et masque
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        mask = mask.resize((width, height), Image.Resampling.LANCZOS)
 
        # Convertir l'image en tenseur [-1, 1]
        img_array = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        img_tensor = (
            torch.from_numpy(img_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device, dtype=text_embeddings.dtype)
        )
 
        # Encoder l'image dans l'espace latent
        init_latents_orig = pipe.vae.encode(img_tensor).latent_dist.sample(generator=generator)
        init_latents_orig = init_latents_orig * pipe.vae.config.scaling_factor
 
        # Convertir le masque en tenseur [0, 1] et redimensionner aux dimensions latentes
        # Le masque est 1 (blanc) où on veut régénérer, 0 (noir) où on conserve
        mask_array = np.array(mask.convert("L")).astype(np.float32) / 255.0
        mask_tensor = (
            torch.from_numpy(mask_array)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device, dtype=text_embeddings.dtype)
        )
        mask_latent = torch.nn.functional.interpolate(
            mask_tensor, size=(height // 8, width // 8), mode="nearest"
        ) # ---- FIN AJOUT V1

    # 4. Boucle de débruitage
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Dupliquer les latents pour le guidage sans classifieur (CFG)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Prédire le bruit résiduel avec le U-Net
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample

        # Guidage sans classifieur (Classifier-free guidance)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Effectuer une étape d'échantillonnage de l'ordonnanceur (scheduler)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # NOTE pour les étudiants (Partie 3 - inpainting) :
        # À ce stade, les latents contiennent l'estimation actuelle de l'image bruitée au pas de temps t.
        # Pour l'inpainting, vous devez intégrer les latents de l'image originale bruitée dans
        # les régions non masquées ici, avant de passer au prochain pas de temps.
                # --- Inpainting : fusionner les régions non masquées ---
        if inpainting_mode and i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            noise = torch.randn_like(init_latents_orig)
            t_prev_tensor = torch.tensor([t_prev.item()], device=device)
            noisy_orig_latents = pipe.scheduler.add_noise(
                init_latents_orig, noise, t_prev_tensor
            )
            # Conserver l'original dans les zones non masquées, le généré dans les zones masquées
            latents = (1 - mask_latent) * noisy_orig_latents + mask_latent * latents

    # 5. Décoder les latents en image
    latents = latents / pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(latents).sample

    # Redimensionner de [-1, 1] vers [0, 1]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)

    # Convertir en PIL.Image
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()
    pil_image = pipe.numpy_to_pil(decoded)[0]

    return pil_image