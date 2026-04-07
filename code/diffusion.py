import torch
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
):
    """
    Boucle de diffusion.
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

    # Concaténer les embeddings conditionnels et inconditionnels pour une seule passe (forward pass)
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

    # 4. Boucle de débruitage
    for t in tqdm(timesteps, desc="Denoising"):
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

    # 5. Décoder les latents en image
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample

    # Redimensionner de [-1, 1] vers [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)

    # Convertir en PIL.Image
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    pil_image = pipe.numpy_to_pil(image)[0]

    return pil_image

