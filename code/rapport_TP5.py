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
    ("mon_image1", "png"),
]

# ---- Section 5 : Remplacement d'objets ----
S5_CASES = [
    ("pyramides", "webp"),
    ("baie_beauport", "avif"),
    ("moulin", "png"),
    ("mon_image2", "png"),
]


# =============================================================================
# Réponses — Remplir les TODO
# =============================================================================

SPECS_ORDINATEUR = """
Processeur Intel Core (Gen 14), famille 6, modèle 198.
Système d'exploitation Windows 11.
Pas de GPU utilisé, le modèle tourne entièrement sur CPU.
"""

TEMPS_GENERATION = """
Environ 13.5 secondes par image avec 10 étapes de débruitage sur CPU.
"""

S1_PROMPT_1 = "a beautiful sunset over the ocean with pink clouds"
S1_PROMPT_2 = "a futuristic city under water with flying cars at night"
S1_PROMPT_3 = "a cozy cabin in a snowy mountain forest"

DISCUSSION_S1 = """
Les trois images générées sont globalement convaincantes malgré l'utilisation du modèle sur CPU.
Le coucher de soleil est celui qui respecte le mieux son prompt, avec des nuages roses très marqués et un beau reflet sur l'océan.
La ville futuriste présente une ambiance cohérente et un style visuel réussi, mais l'effet sous-marin et les voitures volantes restent seulement partiellement visibles.
Le chalet enneigé est très crédible, avec un bon contraste entre la lumière chaude intérieure et l'environnement froid.
Cette image correspond bien à l'idée d'une cabane chaleureuse en forêt enneigée, même si l'aspect montagneux n'est pas vraiment visible.
Dans ces exemples, le modèle semble mieux réussir les scènes naturelles simples que les scènes plus complexes ou très spécifiques.
"""

S2_PROMPT = "a unicorn on the Moon, photorealistic"

REPONSE_CFG_1 = """
Lorsque guidance_scale = 1.0, le guidage supplémentaire du CFG disparaît.
La génération reste liée au prompt, mais sans renforcement notable.
Dans mes résultats, l'image ne ressemble toujours pas clairement à une licorne sur la Lune.
On observe surtout une forme déformée héritée des faibles valeurs de guidage.
Le texte influence donc encore trop peu la scène pour imposer le bon contenu.
Cette valeur reste insuffisante pour obtenir un résultat fidèle ici.
"""

REPONSE_CFG_2 = """
La meilleure adhérence au prompt est observée autour de scale = 10.0.
On distingue clairement une licorne blanche sur un sol lunaire.
La grande Lune en arrière-plan rend la scène très cohérente.
L'image est lisible, détaillée et encore assez photoréaliste.
Le résultat à 7.5 est aussi bon, mais un peu moins convaincant visuellement.
Scale = 10.0 donne ici le meilleur compromis entre fidélité et qualité.
"""

REPONSE_CFG_3 = """
Les artéfacts commencent à apparaître clairement à partir de scale = 15.0.
Les couleurs deviennent plus artificielles et le rendu moins naturel.
À 20.0, l'image est déjà plus stylisée que photoréaliste.
À 40.0, la silhouette se déforme fortement et le fond devient très exagéré.
À 60.0, l'image est presque abstraite et difficilement exploitable.
Un guidage trop élevé dégrade donc rapidement la qualité visuelle.
"""

DISCUSSION_S2 = """
L'expérience montre bien l'effet du facteur de guidage sur la génération.
De 0.0 à 3.0, les images suivent très mal le prompt demandé.
À partir de 5.0, la licorne et le décor lunaire deviennent reconnaissables.
La meilleure zone se situe entre 7.5 et 10.0, avec un bon équilibre global.
Au-delà de 15.0, les artéfacts deviennent visibles et augmentent rapidement.
Ces résultats montrent bien le compromis entre fidélité au texte et stabilité visuelle.
"""

DESCRIPTION_INPAINTING = """
L'implémentation de l'inpainting repose sur la modification de la fonction generate_image dans diffusion.py.
Deux nouveaux paramètres optionnels ont été ajoutés, soit une image d'entrée et un masque binaire.

Avant la boucle de débruitage, l'image et le masque sont redimensionnés à la taille cible (512x512).
L'image est ensuite convertie en tenseur normalisé entre -1 et 1 puis encodée dans l'espace latent par le VAE du pipeline.
Le masque est aussi converti en tenseur [0, 1] et réduit aux dimensions latentes (64x64) avec une interpolation nearest-neighbor.

À chaque étape de débruitage, les régions non masquées sont fusionnées avec les latents de l'image originale bruitée.
La formule utilisée est latents = (1 - mask_latent) * noisy_orig_latents + mask_latent * latents.
Cela permet de garder l'image d'origine hors du masque, tout en générant librement dans la zone masquée.

Le résultat sur chateau_frontenac montre que la personne a bien été retirée.
La reconstruction reste globalement cohérente avec la scène, même si la façade reconstruite présente quelques déformations visibles.
"""

S4_PROMPTS = {
    "graffiti": "a clean brick wall",
    "pisa": "a green field with blue sky and buildings",
    "chateau_frontenac": "a cobblestone street in front of a historic castle",
    "mon_image1": "a bar table with society games on a shelf in the background",
}

DISCUSSION_S4 = """
La suppression de la tour de Pise est le meilleur résultat, car la zone retirée est remplacée assez naturellement par du gazon, un arbre et du ciel, malgré un léger artéfact rectangulaire dans le ciel.
Le Château Frontenac donne un résultat acceptable, la personne disparaît bien, mais la façade reconstruite présente des déformations visibles et un léger flou.
La suppression du graffiti fonctionne partiellement, car le mur est reconstruit, mais on distingue encore des traces du texte et une texture de briques peu réaliste.
Dans l'image personnelle, le résultat est le moins convaincant. La scène devient déformée et le modèle génère des verres et objets imprécis sur la table.
Dans mes essais, les arrière-plans ouverts et peu structurés sont les plus faciles à reconstruire, tandis que les zones riches en détails ou en géométrie sont plus difficiles.
Des prompts précis décrivant clairement l'arrière-plan donnent de meilleurs résultats, mais la taille de l'objet supprimé influence aussi fortement la qualité finale.
"""

S5_PROMPTS = {
    "pyramides": "snow covered pyramid in a winter desert",
    "baie_beauport": "a penguin standing on a sandy beach",
    "moulin": "a giant standing in a field",
    "mon_image2": "a road going to an ice castle in a snowy landscape with mountains in the background",
}

DISCUSSION_S5 = """
Le remplacement sur les pyramides est le plus réussi. La neige s'intègre bien à la scène et le Sphinx reste visible, même si le palmier vert réduit un peu le réalisme.
À la baie de Beauport, le résultat est moins convaincant, car le modèle génère plusieurs formes ressemblant à des pingouins ainsi qu'une structure étrange qui ne correspond pas vraiment au prompt.
Le moulin remplacé par un géant donne une silhouette crédible de loin, mais le résultat reste abstrait et on distingue encore légèrement la forme du moulin d'origine.
Dans l'image personnelle, le décor est bien transformé en paysage montagneux enneigé et le résultat reste cohérent avec la scène.
En revanche, le château de glace demandé n'apparaît pas clairement dans l'image finale.
Dans mes essais, le modèle réussit mieux les changements d'ambiance ou de décor que l'insertion d'objets précis et bien définis.
"""

# ---- Prompts IA ----

# TODO: Copie ici ton premier prompt utilisé avec l'IA
PROMPT_IA_1 = """
Yo
1. Analyse et comprend le travail que je dois faire en Python, soit TP5.pdf
2. Analyse les fichier de codes fournis par le professeur, ainsi que l'arborescence actuelle de mon projet (voir fichier Arborescence.txt).
3. Fait moi un plan détaillé de ce que je vais devoir faire ainsi qu'un plan des différents dossiers et fichiers que je vais devoir créer et
faire (arborescence de mon projet). Toutefois, dans TP5.pdf, n'utilise pas l'emplacement qu'il est mentionné pour mon rapport, utilise plutôt mon Arborescence.txt.
4. Fait moi un plan détaillé (nom des fonctions, noms des variables, et explication de ce que fait chaque fonction). Je veux que tu sois mon tuteur et que tu favories
mon apprentissage, donc ne me donne pas le code complêt. Je veux que te m'aide. S'il te manque d'informations pour en compléter certain, dit moi le et ajoute des zones TODO expliquant quoi
faire. Dit moi les librairies nécessaire au fonctionnement du code. Le seule code que je te permet de me faire au complet sont les fichiers de code qui ne sont
pas directement lié au fonction que je doit faire dans le projet, par exemple le code pour s'occuper d'importer et d'exporter.
5. Dans arbo.txt, tu trouveras l'emplacement des images, ainsi que les dossiers où seront mes images généré.
6. Ne t'occupe pas de la section des questions à répondre dans le rapport, pour l'instant.
7. Finalement, analyse l'énoncé du TP5.pdf pour chaque étapes que je dois faire, dit moi pour quel étapes est-ce que je vais avoir des questions a répondre dans mon rapport.
8. Avant de donner ta réponse, mentionne moi s'il te manque des informations où si quelque chose est pas clair. Si tu as toutes l'informations nécessaire pour
faire ta réponse, analyse la pour etre certain qu'elle répond à mes critères et aux attentes de TP5.pdf. Tu n'as aucune limite de temps pour répondre.
"""

# TODO: Copie ici ton deuxième prompt utilisé avec l'IA
PROMPT_IA_2 = """
Yo
Pour le rapport, voici un exemple de rapport pour le TP4. Peux tu refaire ton code pour le rapport du TP5 en prenant comme exemple celui-ci.
De plus, inidque avec des TODO l'endroite où que je vais devoir ecrire mes reponse. Comme pour le rapport du TP4, ajoute moi 2 zones pour mettre des exemple de prompt
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
