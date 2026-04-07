import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from diffusion import generate_image


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


if __name__ == "__main__":
    main()
