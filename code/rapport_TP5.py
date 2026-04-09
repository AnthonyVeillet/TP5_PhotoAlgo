from __future__ import annotations

from datetime import datetime
from pathlib import Path

# =========================
# Config
# =========================

# Le rapport sera généré à la racine du projet TP5/
OUTPUT_HTML = Path(__file__).resolve().parent.parent / "rapport.html"

TITLE = "🎨 Rapport TP5 — Synthèse d'images génératives et inpainting"
# TODO: Mettre votre nom ici
AUTHOR = "Anthony Veillet"
COURSE_FOOTER = "Photographie algorithmique — TP5 | Modèles de diffusion"

# Chemins relatifs depuis rapport.html (racine TP5/)
OUT = Path("data/dataOutput")
INP = Path("data/dataInput/imgs")

# ---- Section 1 : Réchauffement ----
S1_IMG1 = OUT / "Section 1" / "gen_1.png"
S1_IMG2 = OUT / "Section 1" / "gen_2.png"
S1_IMG3 = OUT / "Section 1" / "gen_3.png"

# ---- Section 2 : CFG ----
S2_GRID = OUT / "Section 2" / "cfg_grid.png"
S2_SCALES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 40.0, 60.0]
S2_IMGS = [OUT / "Section 2" / f"cfg_{s}.png" for s in S2_SCALES]

# ---- Section 3 : Inpainting validation ----
S3_TEST = OUT / "Section 3" / "inpainting_test.png"

# ---- Section 4 : Suppression d'objets ----
S4_CASES = [
    ("graffiti", "png"),
    ("pisa", "png"),
    ("chateau_frontenac", "png"),
    # TODO: Ajouter ton cas personnel ici, par exemple:
    # ("mon_image", "png"),
]

# ---- Section 5 : Remplacement d'objets ----
S5_CASES = [
    ("pyramides", "webp"),
    ("baie_beauport", "avif"),
    ("moulin", "png"),
    # TODO: Ajouter ton cas personnel ici, par exemple:
    # ("mon_image_remplacement", "png"),
]


# =============================================================================
# Réponses — Remplir les TODO
# =============================================================================

# ---- Section 1 : Réchauffement ----

# TODO: Écris les spécifications de ton ordinateur
SPECS_ORDINATEUR = """
TODO: Processeur, RAM, GPU (si applicable), système d'exploitation.
"""

# TODO: Écris le temps de génération observé
TEMPS_GENERATION = """
TODO: Temps de génération pour une image (en secondes).
"""

# TODO: Écris les prompts que tu as utilisés
S1_PROMPT_1 = "TODO: Écris ton premier prompt ici"
S1_PROMPT_2 = "TODO: Écris ton deuxième prompt ici"
S1_PROMPT_3 = "TODO: Écris ton troisième prompt ici"

# TODO: Discussion brève des résultats de la section 1
DISCUSSION_S1 = """
TODO: Discute brièvement de la qualité des images générées avec tes prompts.
"""

# ---- Section 2 : Guidage sans classifieur (CFG) ----

# TODO: Prompt utilisé pour la section 2
S2_PROMPT = "a horse on Mars"

# TODO: Que se passe-t-il lorsque guidance_scale = 1.0 ? Pourquoi ?
REPONSE_CFG_1 = """
TODO: Répondre ici.
"""

# TODO: À quelle échelle de guidage observez-vous la meilleure adhérence au prompt ?
REPONSE_CFG_2 = """
TODO: Répondre ici.
"""

# TODO: À quelle échelle de guidage les artéfacts commencent-ils à apparaître ?
REPONSE_CFG_3 = """
TODO: Répondre ici.
"""

# TODO: Discussion résumant vos observations sur le CFG
DISCUSSION_S2 = """
TODO: Discussion brève résumant vos observations.
"""

# ---- Section 3 : Implémentation de l'inpainting ----

# TODO: Décrivez votre implémentation d'inpainting
DESCRIPTION_INPAINTING = """
TODO: Décrivez les modifications apportées à generate_image pour supporter l'inpainting.
Expliquez les étapes : encodage VAE de l'image, création du masque latent,
fusion à chaque étape de débruitage, etc.
"""

# ---- Section 4 : Suppression d'objets ----

# TODO: Prompts utilisés pour chaque suppression
S4_PROMPTS = {
    "graffiti": "TODO: prompt utilisé",
    "pisa": "TODO: prompt utilisé",
    "chateau_frontenac": "TODO: prompt utilisé",
    # TODO: Ajouter le prompt de ton cas personnel
    # "mon_image": "TODO: prompt utilisé",
}

# TODO: Discussion des résultats de suppression
DISCUSSION_S4 = """
TODO: Discutez des résultats. Quels types d'arrière-plans sont les plus faciles/difficiles
à reconstruire ? Comment la taille de l'objet supprimé affecte-t-elle la qualité ?
Quelles stratégies de prompts fonctionnent le mieux ?
"""

# ---- Section 5 : Remplacement d'objets ----

# TODO: Prompts utilisés pour chaque remplacement
S5_PROMPTS = {
    "pyramides": "TODO: prompt utilisé",
    "baie_beauport": "TODO: prompt utilisé",
    "moulin": "TODO: prompt utilisé",
    # TODO: Ajouter le prompt de ton cas personnel
    # "mon_image_remplacement": "TODO: prompt utilisé",
}

# TODO: Discussion des résultats de remplacement
DISCUSSION_S5 = """
TODO: Discutez de la cohérence de l'éclairage, de l'échelle appropriée
et de la qualité des bords pour chaque remplacement.
"""

# ---- Prompts IA ----

# TODO: Copie ici ton premier prompt utilisé avec l'IA
PROMPT_IA_1 = """
TODO: Coller ici le premier exemple de prompt IA utilisé.
"""

# TODO: Copie ici ton deuxième prompt utilisé avec l'IA
PROMPT_IA_2 = """
TODO: Coller ici le deuxième exemple de prompt IA utilisé.
"""


# =========================
# Helpers HTML
# =========================

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _nl2br(s: str) -> str:
    return s.strip().replace("\n", "<br>\n")


def figure(src: Path, caption: str, max_width: str = "80%") -> str:
    return f"""
    <div class="figure-container">
        <img src="{src.as_posix()}" alt="{_esc(caption)}" data-fullsize="{src.as_posix()}"
             onclick="openLightbox(this)" style="max-width:{max_width};" />
        <p class="figure-caption">{_esc(caption)}</p>
    </div>"""


def trio(a: Path, cap_a: str, b: Path, cap_b: str, c: Path, cap_c: str) -> str:
    """Affiche 3 images côte à côte (original / masque / résultat)."""
    items = ""
    for p, cap in [(a, cap_a), (b, cap_b), (c, cap_c)]:
        items += f"""
        <div class="comparison-image-item" onclick="openLightbox(this.querySelector('img'))">
            <img src="{p.as_posix()}" alt="{_esc(cap)}" data-fullsize="{p.as_posix()}">
            <div class="comparison-image-label">{_esc(cap)}</div>
        </div>"""
    return f'<div class="comparison-images" style="grid-template-columns: repeat(3, 1fr);">{items}</div>'


def grid_images(paths: list, captions: list, cols: int = 3) -> str:
    items = ""
    for p, c in zip(paths, captions):
        items += f"""
        <div class="comparison-image-item" onclick="openLightbox(this.querySelector('img'))">
            <img src="{p.as_posix()}" alt="{_esc(c)}" data-fullsize="{p.as_posix()}">
            <div class="comparison-image-label">{_esc(c)}</div>
        </div>"""
    return f'<div class="comparison-images" style="grid-template-columns: repeat({cols}, 1fr);">{items}</div>'


def text_block(title: str, content: str) -> str:
    return f"""
    <div class="text-block">
        <div class="text-block-title">📝 {_esc(title)}</div>
        <div class="text-content">{_nl2br(content)}</div>
    </div>"""


def decl_ia_block(content: str) -> str:
    return f"""
    <div class="decl-ia">
        <div class="text-block-title">🤖 Déclaration relative à l'IA</div>
        <div class="text-content">{_nl2br(content)}</div>
    </div>"""


def section(title: str, inner: str) -> str:
    return f"""
    <section class="image-section">
        <h2>{_esc(title)}</h2>
        {inner}
    </section>"""


# =========================
# Build sections
# =========================

def build_section1() -> str:
    s = ""
    s += text_block("Spécifications de l'ordinateur", SPECS_ORDINATEUR)
    s += text_block("Temps de génération", TEMPS_GENERATION)

    s += "<h3>Images générées</h3>"

    s += f"<h4>Prompt 1 : « {_esc(S1_PROMPT_1)} »</h4>"
    s += figure(S1_IMG1, f"Prompt : {S1_PROMPT_1}")

    s += f"<h4>Prompt 2 : « {_esc(S1_PROMPT_2)} »</h4>"
    s += figure(S1_IMG2, f"Prompt : {S1_PROMPT_2}")

    s += f"<h4>Prompt 3 : « {_esc(S1_PROMPT_3)} »</h4>"
    s += figure(S1_IMG3, f"Prompt : {S1_PROMPT_3}")

    s += text_block("Discussion", DISCUSSION_S1)
    return s


def build_section2() -> str:
    s = ""
    s += f"<p style='color:#a0a0a0;'>Prompt utilisé : <b>« {_esc(S2_PROMPT)} »</b></p>"

    s += "<h3>Grille comparative des échelles de guidage</h3>"
    s += figure(S2_GRID, "Grille CFG — toutes les échelles de guidage", "95%")

    s += "<h3>Images individuelles</h3>"
    s += grid_images(S2_IMGS, [f"scale = {s}" for s in S2_SCALES], cols=4)

    s += "<hr class='soft-hr' />"
    s += "<h3>Questions d'analyse</h3>"
    s += text_block("Que se passe-t-il lorsque guidance_scale = 1.0 ? Pourquoi ?", REPONSE_CFG_1)
    s += text_block("Meilleure adhérence au prompt — à quelle échelle ?", REPONSE_CFG_2)
    s += text_block("Apparition des artéfacts — à quelle échelle ?", REPONSE_CFG_3)
    s += text_block("Discussion", DISCUSSION_S2)
    return s


def build_section3() -> str:
    s = ""
    s += text_block("Description de l'implémentation", DESCRIPTION_INPAINTING)

    s += "<h3>Validation</h3>"
    s += figure(S3_TEST, "Test d'inpainting — chateau_frontenac")
    return s


def build_section4() -> str:
    s = ""

    for name, ext in S4_CASES:
        prompt = S4_PROMPTS.get(name, "")
        orig = INP / f"{name}.{ext}"
        mask = INP / f"{name}_mask.png"
        result = OUT / "Section 4" / f"{name}_result.png"

        s += f"<h3>{name}</h3>"
        s += f"<p style='color:#a0a0a0;'>Prompt : <b>« {_esc(prompt)} »</b></p>"
        s += trio(orig, "Original", mask, "Masque", result, "Résultat")
        s += "<hr class='soft-hr' />"

    s += text_block("Discussion — Suppression d'objets", DISCUSSION_S4)
    return s


def build_section5() -> str:
    s = ""

    for name, ext in S5_CASES:
        prompt = S5_PROMPTS.get(name, "")
        orig = INP / f"{name}.{ext}"
        mask = INP / f"{name}_mask.png"
        result = OUT / "Section 5" / f"{name}_result.png"

        s += f"<h3>{name}</h3>"
        s += f"<p style='color:#a0a0a0;'>Prompt : <b>« {_esc(prompt)} »</b></p>"
        s += trio(orig, "Original", mask, "Masque", result, "Résultat")
        s += "<hr class='soft-hr' />"

    s += text_block("Discussion — Remplacement d'objets", DISCUSSION_S5)
    return s


def build_prompts_ia() -> str:
    s = ""
    s += text_block("Exemple de prompt 1", PROMPT_IA_1)
    s += "<hr class='soft-hr' />"
    s += text_block("Exemple de prompt 2", PROMPT_IA_2)
    return s


# =========================
# HTML Template
# =========================

def build_html() -> str:
    now = datetime.now().strftime("%d %B %Y à %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{_esc(TITLE)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');
    * {{ box-sizing: border-box; }}

    body {{
      font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 0;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      color: #e8e8e8;
      min-height: 100vh;
    }}

    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 30px 20px;
    }}

    header {{
      text-align: center;
      padding: 40px 0;
      border-bottom: 2px solid rgba(255,255,255,0.1);
      margin-bottom: 40px;
    }}

    h1 {{
      font-size: 2.5em;
      font-weight: 700;
      margin: 0 0 10px 0;
      text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }}

    .author {{
      font-size: 1.15em;
      color: #b0b0b0;
      margin-bottom: 6px;
    }}

    .date-badge {{
      display: inline-block;
      background: rgba(255,255,255,0.1);
      padding: 8px 20px;
      border-radius: 20px;
      margin-top: 12px;
      font-size: 0.9em;
      color: #b0b0b0;
    }}

    .image-section {{
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(10px);
      border-radius: 16px;
      padding: 30px;
      margin-bottom: 40px;
      border: 1px solid rgba(255,255,255,0.1);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }}

    .image-section h2 {{
      color: #778da9;
      font-size: 1.6em;
      margin: 0 0 25px 0;
      padding-bottom: 15px;
      border-bottom: 2px solid rgba(119, 141, 169, 0.25);
    }}

    h3 {{ color: #e0e1dd; font-size: 1.3em; margin: 26px 0 14px 0; }}
    h4 {{ margin: 18px 0 10px 0; color: #dbe2ef; }}

    .figure-container {{
      text-align: center;
      margin: 15px 0;
      padding: 15px;
      background: rgba(0,0,0,0.2);
      border-radius: 12px;
    }}

    .figure-container img {{
      max-width: 100%;
      max-height: 600px;
      border-radius: 8px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
    }}

    .figure-container img:hover {{
      transform: scale(1.015);
      box-shadow: 0 6px 30px rgba(0,0,0,0.5);
    }}

    .figure-caption {{
      margin-top: 10px;
      font-style: italic;
      color: #a0a0a0;
      font-size: 0.9em;
    }}

    .comparison-images {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin: 10px 0;
    }}

    .comparison-image-item {{
      position: relative;
      border-radius: 10px;
      overflow: hidden;
      background: rgba(0,0,0,0.2);
      cursor: pointer;
      transition: transform 0.15s, box-shadow 0.15s;
    }}

    .comparison-image-item:hover {{
      transform: translateY(-3px);
      box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    }}

    .comparison-image-item img {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .comparison-image-label {{
      position: absolute;
      bottom: 0; left: 0; right: 0;
      background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
      color: #fff;
      padding: 12px 8px 8px;
      font-size: 0.85em;
      text-align: center;
      font-weight: 500;
    }}

    .text-block, .decl-ia {{
      background: rgba(0,0,0,0.25);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 18px;
      margin: 18px 0 0 0;
    }}

    .decl-ia {{ border-left: 4px solid #778da9; }}

    .text-block-title {{
      font-weight: 700;
      color: #cbd5e1;
      margin-bottom: 10px;
    }}

    .text-content {{
      color: #d0d0d0;
      line-height: 1.7;
      font-size: 1em;
    }}

    .soft-hr {{
      border: none;
      height: 1px;
      background: rgba(255,255,255,0.12);
      margin: 22px 0;
    }}

    footer {{
      text-align: center;
      padding: 30px;
      color: #777;
      font-size: 0.95em;
    }}

    .lightbox {{
      display: none;
      position: fixed;
      z-index: 9999;
      left: 0; top: 0;
      width: 100%; height: 100%;
      background-color: rgba(0,0,0,0.92);
      animation: fadeIn 0.2s;
    }}

    .lightbox.active {{
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .lightbox-content {{
      max-width: 95vw;
      max-height: 95vh;
      padding: 18px;
    }}

    .lightbox-content img {{
      max-width: 100%;
      max-height: 95vh;
      object-fit: contain;
      border-radius: 10px;
      box-shadow: 0 8px 40px rgba(0,0,0,0.8);
    }}

    .lightbox-close {{
      position: absolute;
      top: 16px; right: 30px;
      color: #fff;
      font-size: 44px;
      font-weight: bold;
      cursor: pointer;
      user-select: none;
    }}

    .lightbox-close:hover {{ color: #ffc107; }}

    @keyframes fadeIn {{
      from {{ opacity: 0; }}
      to {{ opacity: 1; }}
    }}

    @media (max-width: 900px) {{
      .comparison-images {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>

<body>
  <div id="lightbox" class="lightbox" onclick="closeLightbox(event)">
    <span class="lightbox-close">&times;</span>
    <div class="lightbox-content">
      <img id="lightbox-img" src="" alt="">
    </div>
  </div>

  <div class="container">
    <header>
      <h1>{_esc(TITLE)}</h1>
      <div class="author">{_esc(AUTHOR)}</div>
      <div class="date-badge">Généré le {now}</div>
    </header>

    {section("1. Réchauffement (15%)", build_section1())}
    {section("2. Guidage sans classifieur — CFG (15%)", build_section2())}
    {section("3. Implémentation de l'inpainting (30%)", build_section3())}
    {section("4. Suppression d'objets (15%)", build_section4())}
    {section("5. Remplacement d'objets (15%)", build_section5())}
    {section("Annexe — Exemples de prompts IA utilisés", build_prompts_ia())}

    <footer>
      <p>{_esc(COURSE_FOOTER)}</p>
    </footer>
  </div>

  <script>
    function openLightbox(img) {{
      const lightbox = document.getElementById('lightbox');
      const lightboxImg = document.getElementById('lightbox-img');
      lightboxImg.src = img.getAttribute('data-fullsize') || img.src;
      lightbox.classList.add('active');
      document.body.style.overflow = 'hidden';
    }}

    function closeLightbox(event) {{
      const lightbox = document.getElementById('lightbox');
      if (event.target === lightbox || event.target.classList.contains('lightbox-close')) {{
        lightbox.classList.remove('active');
        document.body.style.overflow = 'auto';
      }}
    }}

    document.addEventListener('keydown', function(event) {{
      if (event.key === 'Escape') {{
        document.getElementById('lightbox').classList.remove('active');
        document.body.style.overflow = 'auto';
      }}
    }});
  </script>
</body>
</html>"""
    return html


def main() -> None:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"[OK] Rapport généré: {OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
